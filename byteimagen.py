class ByteImage(nn.Module):
    def __init__(self):
        self.byteformer = Byteformer()
        self.imagen = Imagen()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """

        Args:
            x: Bytes tensor []
            y: Image tensor (RGB) (GT) []

        Returns:

        """
        text_embedding = self.byteformer(x)
        loss = self.imagen(images=y, text_embeds=text_embedding, unet_number=i)
        return loss



model = ByteFormer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in epochs:
    loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

