import torch
from torch import nn
model=nn.TransformerEncoderLayer(d_model=256,nhead=8)
src=torch.randn(10,32,256,dtype=torch.float32)

output=model(src)



#%%

model2=nn.TransformerDecoderLayer(d_model=256,nhead=8)
mem=torch.randn(10,32,256)
tgt=torch.randn(20,32,256)
out2=model2(tgt,mem)
print(out2.shape)
#%%
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
Transformer_encoder=nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=6)
src=torch.randn(10,32,512)
test=Transformer_encoder(src)

#%%
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)