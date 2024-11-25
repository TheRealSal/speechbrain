
import torch
import numpy
import math

class SepReformer(torch.nn.Module):
    def __init__(self,
                 num_stages: int,
                 num_spks: int,
                 module_audio_enc: dict,
                 module_feature_projector: dict,
                 module_separator: dict,
                 module_output_layer: dict,
                 module_audio_dec: dict):
        super().__init__()
        self.num_stages = num_stages
        self.num_spks = num_spks
        self.audio_encoder = AudioEncoder(**module_audio_enc)
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)

        # Aux_loss
        self.out_layer_bn = torch.nn.ModuleList([])
        self.decoder_bn = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.out_layer_bn.append(OutputLayer(**module_output_layer, masking=True))
            self.decoder_bn.append(AudioDecoder(**module_audio_dec))

    def forward(self, x):
        encoder_output = self.audio_encoder(x)
        projected_feature = self.feature_projector(encoder_output)
        last_stage_output, each_stage_outputs = self.separator(projected_feature)
        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        each_spk_output = [out_layer_output[idx] for idx in range(self.num_spks)]
        audio = [self.audio_decoder(each_spk_output[idx]) for idx in range(self.num_spks)]

        return audio


class AudioEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int, bias: bool):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups,
            bias=bias)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(x,
                                                                                dim=1)  # [T] - >[1, T] OR [B, T] -> [B, 1, T]
        x = self.conv1d(x)
        x = self.gelu(x)
        return x


class FeatureProjector(torch.nn.Module):
    def __init__(self, num_channels: int, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-8)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.conv1d(x)
        return x


class Separator(torch.nn.Module):
    def __init__(self, num_stages: int, relative_positional_encoding: dict, enc_stage: dict, spk_split_stage: dict,
                 simple_fusion: dict, dec_stage: dict):
        super().__init__()

        class RelativePositionalEncoding(torch.nn.Module):
            def __init__(self, in_channels: int, num_heads: int, maxlen: int, embed_v=False):
                super().__init__()
                self.in_channels = in_channels
                self.num_heads = num_heads
                self.embedding_dim = self.in_channels // self.num_heads
                self.maxlen = maxlen
                self.pe_k = torch.nn.Embedding(num_embeddings=2 * maxlen, embedding_dim=self.embedding_dim)
                self.pe_v = torch.nn.Embedding(num_embeddings=2 * maxlen,
                                               embedding_dim=self.embedding_dim) if embed_v else None

            def forward(self, pos_seq: torch.Tensor):
                pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
                pos_seq += self.maxlen
                pe_k_output = self.pe_k(pos_seq)
                pe_v_output = self.pe_v(pos_seq) if self.pe_v is not None else None
                return pe_k_output, pe_v_output

        class SepEncStage(torch.nn.Module):
            def __init__(self, global_blocks: dict, local_blocks: dict, down_conv_layer: dict, down_conv=True):
                super().__init__()

                class DownConvLayer(torch.nn.Module):
                    def __init__(self, in_channels: int, samp_kernel_size: int):
                        """Construct an EncoderLayer object."""
                        super().__init__()
                        self.down_conv = torch.nn.Conv1d(
                            in_channels=in_channels, out_channels=in_channels, kernel_size=samp_kernel_size, stride=2,
                            padding=(samp_kernel_size - 1) // 2, groups=in_channels)
                        self.BN = torch.nn.BatchNorm1d(num_features=in_channels)
                        self.gelu = torch.nn.GELU()

                    def forward(self, x: torch.Tensor):
                        x = x.permute([0, 2, 1])
                        x = self.down_conv(x)
                        x = self.BN(x)
                        x = self.gelu(x)
                        x = x.permute([0, 2, 1])
                        return x

                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)

                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)

                self.downconv = DownConvLayer(**down_conv_layer) if down_conv == True else None

            def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
                '''
                x: [B, N, T]
                '''
                x = self.g_block_1(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()

                x = self.g_block_2(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_2(x)
                x = x.permute(0, 2, 1).contiguous()

                skip = x
                if self.downconv:
                    x = x.permute(0, 2, 1).contiguous()
                    x = self.downconv(x)
                    x = x.permute(0, 2, 1).contiguous()
                # [BK, S, N]
                return x, skip

        class SpkSplitStage(torch.nn.Module):
            def __init__(self, in_channels: int, num_spks: int):
                super().__init__()
                self.linear = torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels, 4 * in_channels * num_spks, kernel_size=1),
                    torch.nn.GLU(dim=-2),
                    torch.nn.Conv1d(2 * in_channels * num_spks, in_channels * num_spks, kernel_size=1))
                self.norm = torch.nn.GroupNorm(1, in_channels, eps=1e-8)
                self.num_spks = num_spks

            def forward(self, x: torch.Tensor):
                x = self.linear(x)
                B, _, T = x.shape
                x = x.view(B * self.num_spks, -1, T).contiguous()
                x = self.norm(x)
                return x

        class SepDecStage(torch.nn.Module):
            def __init__(self, num_spks: int, global_blocks: dict, local_blocks: dict, spk_attention: dict):
                super().__init__()

                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                self.spk_attn_1 = SpkAttention(**spk_attention)

                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)
                self.spk_attn_2 = SpkAttention(**spk_attention)

                self.g_block_3 = GlobalBlock(**global_blocks)
                self.l_block_3 = LocalBlock(**local_blocks)
                self.spk_attn_3 = SpkAttention(**spk_attention)

                self.num_spk = num_spks

            def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
                '''
                x: [B, N, T]
                '''
                # [BS, K, H]
                x = self.g_block_1(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_1(x, self.num_spk)

                x = self.g_block_2(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_2(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_2(x, self.num_spk)

                x = self.g_block_3(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_3(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_3(x, self.num_spk)

                skip = x

                return x, skip

        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(**relative_positional_encoding)

        # Temporal Contracting Part
        self.enc_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.enc_stages.append(SepEncStage(**enc_stage, down_conv=True))

        self.bottleneck_G = SepEncStage(**enc_stage, down_conv=False)
        self.spk_split_block = SpkSplitStage(**spk_split_stage)

        # Temporal Expanding Part
        self.simple_fusion = torch.nn.ModuleList([])
        self.dec_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.simple_fusion.append(torch.nn.Conv1d(in_channels=simple_fusion['out_channels'] * 2,
                                                      out_channels=simple_fusion['out_channels'], kernel_size=1))
            self.dec_stages.append(SepDecStage(**dec_stage))

    def forward(self, input: torch.Tensor):
        '''input: [B, N, L]'''
        # feature projection
        x, _ = self.pad_signal(input)
        len_x = x.shape[-1]
        # Temporal Contracting Part
        pos_seq = torch.arange(0, len_x // 2 ** self.num_stages).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)
        skip = []
        for idx in range(self.num_stages):
            x, skip_ = self.enc_stages[idx](x, pos_k)
            skip_ = self.spk_split_block(skip_)
            skip.append(skip_)
        x, _ = self.bottleneck_G(x, pos_k)
        x = self.spk_split_block(x)  # B, 2F, T

        each_stage_outputs = []
        # Temporal Expanding Part
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = torch.nn.functional.upsample(x, skip[idx_en].shape[-1])
            x = torch.cat([x, skip[idx_en]], dim=1)
            x = self.simple_fusion[idx](x)
            x, _ = self.dec_stages[idx](x, pos_k)

        last_stage_output = x
        return last_stage_output, each_stage_outputs

    def pad_signal(self, input: torch.Tensor):
        #  (B, T) or (B, 1, T)
        if input.dim() == 1:
            input = input.unsqueeze(0)
        elif input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        elif input.dim() == 2:
            input = input.unsqueeze(1)
        L = 2 ** self.num_stages
        batch_size = input.size(0)
        ndim = input.size(1)
        nframe = input.size(2)
        padded_len = (nframe // L + 1) * L
        rest = 0 if nframe % L == 0 else padded_len - nframe
        if rest > 0:
            pad = torch.autograd.Variable(torch.zeros(batch_size, ndim, rest)).type(input.type()).to(input.device)
            input = torch.cat([input, pad], dim=-1)
        return input, rest


class OutputLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_spks: int, masking: bool = False):
        super().__init__()
        # feature expansion back
        self.masking = masking
        self.spe_block = Masking(in_channels, Activation_mask="ReLU", concat_opt=None)
        self.num_spks = num_spks
        self.end_conv1x1 = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 4 * out_channels),
            torch.nn.GLU(),
            torch.nn.Linear(2 * out_channels, in_channels))

    def forward(self, x: torch.Tensor, input: torch.Tensor):
        x = x[..., :input.shape[-1]]
        x = x.permute([0, 2, 1])
        x = self.end_conv1x1(x)
        x = x.permute([0, 2, 1])
        B, N, L = x.shape
        B = B // self.num_spks

        if self.masking:
            input = input.expand(self.num_spks, B, N, L).transpose(0, 1).contiguous()
            input = input.view(B * self.num_spks, N, L)
            x = self.spe_block(x, input)

        x = x.view(B, self.num_spks, N, L)
        # [spks, B, N, L]
        x = x.transpose(0, 1)
        return x


class AudioDecoder(torch.nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # x: [B, N, L]
        if x.dim() not in [2, 3]: raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        x = torch.squeeze(x, dim=1) if torch.squeeze(x).dim() == 1 else torch.squeeze(x)
        return x

class LayerScale(torch.nn.Module):
    def __init__(self, dims, input_size, Layer_scale_init=1.0e-5):
        super().__init__()
        if dims == 1:
            self.layer_scale = torch.nn.Parameter(torch.ones(input_size) * Layer_scale_init, requires_grad=True)
        elif dims == 2:
            self.layer_scale = torch.nn.Parameter(torch.ones(1, input_size) * Layer_scale_init, requires_grad=True)
        elif dims == 3:
            self.layer_scale = torch.nn.Parameter(torch.ones(1, 1, input_size) * Layer_scale_init, requires_grad=True)

    def forward(self, x):
        return x * self.layer_scale


class Masking(torch.nn.Module):
    def __init__(self, input_dim, Activation_mask='Sigmoid', **options):
        super(Masking, self).__init__()

        self.options = options
        if self.options['concat_opt']:
            self.pw_conv = torch.nn.Conv1d(input_dim * 2, input_dim, 1, stride=1, padding=0)

        if Activation_mask == 'Sigmoid':
            self.gate_act = torch.nn.Sigmoid()
        elif Activation_mask == 'ReLU':
            self.gate_act = torch.nn.ReLU()

    def forward(self, x, skip):

        if self.options['concat_opt']:
            y = torch.cat([x, skip], dim=-2)
            y = self.pw_conv(y)
        else:
            y = x
        y = self.gate_act(y) * skip

        return y


class GCFN(torch.nn.Module):
    def __init__(self, in_channels, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.LayerNorm(in_channels),
            torch.nn.Linear(in_channels, in_channels * 6))
        self.depthwise = torch.nn.Conv1d(in_channels * 6, in_channels * 6, 3, padding=1, groups=in_channels * 6)
        self.net2 = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_channels * 3, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)

    def forward(self, x):
        y = self.net1(x)
        y = y.permute(0, 2, 1).contiguous()
        y = self.depthwise(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.net2(y)
        return x + self.Layer_scale(y)


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention layer.
        :param int n_head: the number of head s
        :param int n_feat: the number of features
        :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head: int, in_channels: int, dropout_rate: float, Layer_scale_init=1.0e-5):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head  # We assume d_v always equals d_k
        self.h = n_head
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear_q = torch.nn.Linear(in_channels, in_channels)
        self.linear_k = torch.nn.Linear(in_channels, in_channels)
        self.linear_v = torch.nn.Linear(in_channels, in_channels)
        self.linear_out = torch.nn.Linear(in_channels, in_channels)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)

    def forward(self, x, pos_k, mask):
        """
        Compute 'Scaled Dot Product Attention'.
            :param torch.Tensor mask: (batch, time1, time2)
            :param torch.nn.Dropout dropout:
            :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
            weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)  # (b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)  # (b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0, 1)
        if pos_k is not None:
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.Layer_scale(self.dropout(self.linear_out(x)))  # (batch, time1, d_model)


class EGA(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict({
            'self_attn': MultiHeadAttention(
                n_head=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate),
            'linear': torch.nn.Sequential(
                torch.nn.LayerNorm(normalized_shape=in_channels),
                torch.nn.Linear(in_features=in_channels, out_features=in_channels),
                torch.nn.Sigmoid())
        })

    def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
        """
        Compute encoded features.
            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
            :param torch.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        down_len = pos_k.shape[0]
        x_down = torch.nn.functional.adaptive_avg_pool1d(input=x, output_size=down_len)
        x = x.permute([0, 2, 1])
        x_down = x_down.permute([0, 2, 1])
        x_down = self.block['self_attn'](x_down, pos_k, None)
        x_down = x_down.permute([0, 2, 1])
        x_downup = torch.nn.functional.upsample(input=x_down, size=x.shape[1])
        x_downup = x_downup.permute([0, 2, 1])
        x = x + self.block['linear'](x) * x_downup

        return x


class CLA(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear1 = torch.nn.Linear(in_channels, in_channels * 2)
        self.GLU = torch.nn.GLU()
        self.dw_conv_1d = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels)
        self.linear2 = torch.nn.Linear(in_channels, 2 * in_channels)
        self.BN = torch.nn.BatchNorm1d(2 * in_channels)
        self.linear3 = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(2 * in_channels, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)

    def forward(self, x):
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.GLU(y)
        y = y.permute([0, 2, 1])  # B, F, T
        y = self.dw_conv_1d(y)
        y = y.permute(0, 2, 1)  # B, T, 2F
        y = self.linear2(y)
        y = y.permute(0, 2, 1)  # B, T, 2F
        y = self.BN(y)
        y = y.permute(0, 2, 1)  # B, T, 2F
        y = self.linear3(y)

        return x + self.Layer_scale(y)


class GlobalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict({
            'ega': EGA(
                num_mha_heads=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate),
            'gcfn': GCFN(in_channels=in_channels, dropout_rate=dropout_rate)
        })

    def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
        """
        Compute encoded features.
            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
            :param torch.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x = self.block['ega'](x, pos_k)
        x = self.block['gcfn'](x)
        x = x.permute([0, 2, 1])

        return x


class LocalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict({
            'cla': CLA(in_channels, kernel_size, dropout_rate),
            'gcfn': GCFN(in_channels, dropout_rate)
        })

    def forward(self, x: torch.Tensor):
        x = self.block['cla'](x)
        x = self.block['gcfn'](x)

        return x


class SpkAttention(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate)
        self.feed_forward = GCFN(in_channels=in_channels, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, num_spk: int):
        """
        Compute encoded features.
            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
            :param torch.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        B, F, T = x.shape
        x = x.view(B // num_spk, num_spk, F, T).contiguous()
        x = x.permute([0, 3, 1, 2]).contiguous()
        x = x.view(-1, num_spk, F).contiguous()
        x = x + self.self_attn(x, None, None)
        x = x.view(B // num_spk, T, num_spk, F).contiguous()
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = x.view(B, F, T).contiguous()
        x = x.permute([0, 2, 1])
        x = self.feed_forward(x)
        x = x.permute([0, 2, 1])

        return x