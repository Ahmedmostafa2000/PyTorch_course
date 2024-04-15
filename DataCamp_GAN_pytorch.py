def gen_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True)
    )

def disc_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LeakyReLU(0.2)
    )

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        # Define generator block
        self.generator = nn.Sequential(
            gen_block(in_dim, 256),
            gen_block(256, 512),
            gen_block(512, 1024),
            # Add linear layer
            nn.Linear(1024,in_dim),
            # Add activation
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass input through generator
        return seld.generator()

class Discriminator(nn.Module):
    def __init__(self, im_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            disc_block(im_dim, 1024),
            disc_block(1024, 512),
            # Define last discriminator block
            disc_block(512,254),
            # Add a linear layer
            nn.Linear(256,1),
        )

    def forward(self, x):
        # Define the forward method
        self.disc(x)

class DCGenerator(nn.Module):
    def __init__(self, in_dim, kernel_size=4, stride=2):
        super(DCGenerator, self).__init__()
        self.in_dim = in_dim
        self.gen = nn.Sequential(
            dc_gen_block(in_dim, 1024, kernel_size, stride),
            dc_gen_block(1024, 512, kernel_size, stride),
            # Add last generator block
            dc_gen_block(512,256,kernel_size,stride),
            # Add transposed convolution
            nn.ConvTranspose2d(256, 3, kernel_size, stride=stride),
            # Add tanh activation
            nn.tanh()
        )

    def forward(self, x):
        x = x.view(len(x), self.in_dim, 1, 1)
        return self.gen(x)

class DCDiscriminator(nn.Module):
    def __init__(self, kernel_size=4, stride=2):
        super(DCDiscriminator, self).__init__()
        self.disc = nn.Sequential(
            # Add first discriminator block
            dc_disc_block(3, 512, kernel_size, stride),
            dc_disc_block(512, 1024, kernel_size, stride),
            # Add a convolution
            nn.Conv2d(1024, 1, kernel_size, stride=stride),
        )

    def forward(self, x):
        # Pass input through sequential block
        x = self.disc(x)
        return x.view(len(x), -1)


def gen_loss(gen, disc, criterion, num_images, z_dim):
    # Define random noise
    noise = torch.radn(num_images, z_dim)
    # Generate fake image
    fake = gen(noise)
    # Get discriminator's prediction on the fake image
    disc_pred = disc(fake)
    # Compute generator loss
    criterion = nn.BCEWithLogitsLoss()
    gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))
    return gen_loss

def disc_loss(gen, disc, real, num_images, z_dim):
    criterion = nn.BCEWithLogitsLoss()
    noise = torch.randn(num_images, z_dim)
    fake = gen(noise)
    # Get discriminator's predictions for fake images
    disc_pred_fake = disc(fake)
    # Calculate the fake loss component
    fake_loss = criterion(disc_pred_fake,torch.zeros_like(disc_pred_fake))
    # Get discriminator's predictions for real images
    disc_pred_real = disc(real)
    # Calculate the real loss component
    real_loss = criterion(disc_pred_real,torch.ones_like(disc_pred_real))
    disc_loss = (real_loss + fake_loss) / 2
    return disc_loss