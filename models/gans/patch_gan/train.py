import torch
from torch import nn
from torch.autograd import Variable

from .discriminator import Discriminator



adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()


def generator_loss(generated_image, target_img, G, real_target):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    #print(gen_loss)
    return gen_total_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss


def train(channels, width, height, input_img, target_img, generator, discriminator, D_optimizer, G_optimizer, device):
    D_optimizer.zero_grad()
    input_img = input_img.to(device)
    target_img = target_img.to(device)

    # ground truth labels real and fake
    real_target = Variable(torch.ones(input_img.size(0), channels, height, width).to(device))
    fake_target = Variable(torch.zeros(input_img.size(0), channels, height, width).to(device))

    # generator forward pass
    generated_image = generator(input_img)

    # train discriminator with fake/generated images
    disc_inp_fake = torch.cat((input_img, generated_image), 1)

    D_fake = discriminator(disc_inp_fake.detach())

    D_fake_loss = discriminator_loss(D_fake, fake_target)

    # train discriminator with real images
    disc_inp_real = torch.cat((input_img, target_img), 1)

    D_real = discriminator(disc_inp_real)
    D_real_loss = discriminator_loss(D_real, real_target)

    # average discriminator loss
    D_total_loss = (D_real_loss + D_fake_loss) / 2
    # compute gradients and run optimizer step
    D_total_loss.backward()
    D_optimizer.step()

    # Train generator with real labels
    G_optimizer.zero_grad()
    fake_gen = torch.cat((input_img, generated_image), 1)
    G = discriminator(fake_gen)
    G_loss = generator_loss(generated_image, target_img, G, real_target)
    # compute gradients and run optimizer step
    G_loss.backward()
    G_optimizer.step()

    return D_real, G_loss