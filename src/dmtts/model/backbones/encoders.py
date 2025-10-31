import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from dmtts.model import modules
from dmtts.model import attentions
from dmtts.model import commons


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
        num_languages=None, #
        num_tones=None,
        convnext_layers=4,  # ConvNeXtV2 block 개수
        convnext_mult=2,
        max_pos=2048,
        lang_list=None,
    ):
        super().__init__()
        if num_languages is None:
            from dmtts.model.text.symbols import get_language_id
            _, num_languages = get_language_id(lang_list)

        if num_tones is None:
            from dmtts.model.text.symbols import get_tone_id
            _, num_tones = get_tone_id(lang_list)

        self.n_vocab = n_vocab # phoneme vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # text embedding
        self.text_embed = nn.Embedding(n_vocab, hidden_channels) # hidden_channels == text_dim
        nn.init.normal_(self.text_embed.weight, 0.0, hidden_channels**-0.5)

        # tone embedding
        self.tone_embed = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_embed.weight, 0.0, hidden_channels**-0.5)

        # concat -> projection
        self.proj_in = nn.Linear(2*hidden_channels, hidden_channels)

        # conv feature extractor 
        if convnext_layers > 0 :
            self.text_blocks = nn.Sequential(
                *[modules.ConvNeXtV2Block(hidden_channels, hidden_channels*convnext_mult) for _ in range(convnext_layers)]
            )
        else:
            self.text_blocks = nn.Identity()
       
        # sinusoidal positional embedding
        self.pos_emb = modules.SinusPositionEmbedding(hidden_channels)
        self.max_pos = max_pos
        # attention encoder
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        #self.proj = nn.Linear(hidden_channels, out_channels *2)

    #def forward(self, x, x_lengths, tone, language, bert, ja_bert, g=None):
    def forward(self, x, x_lengths, tone, g=None):

        #print("#### TextEncoder ###### 1 ####### Forward #######")
        #print(f"tone shape: {tone.shape}, dtype: {tone.dtype}, device: {tone.device}")
        #print(f"tone min: {tone.min().item()}, max: {tone.max().item()}, "
        #    f"num_tones allowed: {self.tone_embed.num_embeddings}")
        B, T = x.size()

        #print("##### DEBUG: text_embed input check #####")
        #print(f"x shape       : {x.shape}, dtype={x.dtype}, device={x.device}")
        #print(f"x min / max   : {x.min().item()} / {x.max().item()}")
        #print(f"n_vocab       : {self.text_embed.num_embeddings}")
        #print(f"unique tokens : {torch.unique(x)}")
        #exit()


        # Phoneme
        x_ph = self.text_embed(x) * math.sqrt(self.hidden_channels) # [B, T, H]
        #print(f"x_ph    :{x_ph.size()}")
        # Tone
        x_tn = self.tone_embed(tone) * math.sqrt(self.hidden_channels) # [B, T, H]
        #print(f"x_tn    :{x_tn.size()}")
        # Concat + Linear Projection

        #print("#### TextEncoder ###### 2 ####### Forward #######")
        x = torch.cat([x_ph, x_tn], dim = -1) # [B, T, 2H]
        #print(f"concat  :{x.size()}")
        #print("#### TextEncoder ###### 3 ####### Forward #######")


        ###### 문제가 되는 부분 #####

        #print(f"concat dtype: {x.dtype}, device: {x.device}")
        #print(f"proj_in.weight dtype: {self.proj_in.weight.dtype}, device: {self.proj_in.weight.device}")



        x = self.proj_in(x) # [B, T, H]



        #print("#### TextEncoder ###### 4 ####### Forward #######")
        pos_idx = torch.arange(T, device=x.device)  # [T]
        pos_emb = self.pos_emb(pos_idx)             # [T, H]
        pos_emb = pos_emb.unsqueeze(0).expand(B, T, -1)  # [B, T, H]
        #print("#### TextEncoder ###### 5 ####### Forward #######")
        x = x + pos_emb

        #print("#### TextEncoder ###### 6 ####### Forward #######")
        # ConvNeXtV2 stacks
        x = self.text_blocks(x) # [B, T, H]
        x = x.transpose(1,2) # [B, H, T]
        #print("#### TextEncoder ###### 7 ####### Forward #######")
        x_mask = torch.unsqueeze(
            (torch.arange(x.size(2), device=x.device)[None, :] < x_lengths[:, None]), 1
        ).to(x.dtype)

        #print("#### TextEncoder ###### 8 ####### Forward #######")
        x = self.encoder(x * x_mask, x_mask, g=g)
        # projection → mean & logvar
        stats = self.proj(x) * x_mask         # [B, 2*out_channels, T]
        #print("#### TextEncoder ###### 9 ####### Forward #######")
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, tau=1.0):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0, layernorm=False):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)]) # noqa: E501

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)
        if layernorm:
            self.layernorm = nn.LayerNorm(self.spec_channels)
            print('[Ref Enc]: using layer norm')
        else:
            self.layernorm = None

    def forward(self, inputs, mask=None):
        N = inputs.size(0)

        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        if self.layernorm is not None:
            out = self.layernorm(out)

        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L    