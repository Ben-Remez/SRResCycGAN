import lpips
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage

import SRDataset
from components.LR_Generator import *
from components.SR_Generator import *
from utils import *
from Losses import *
from components.Discriminator import *

# ======= Config Parameters =======

# Training Config
load_nets = False
changing_LR = False
freeze_fine_tune_layers = False  # When using fine tune also set only train generator to True!
only_train_generator = False
save_final_model_weights = False
save_model_weights_per_epoch = False
plot_loss_graphs = False
run_training = False

# Train Paths
lr_dir = '/data/LR'
hr_dir = '/data/HR'

# Test Config
run_test = False
plot_metrics_graphs = False
num_models = 30

# Test Paths
lr_test_dir = '/test/LR'
hr_test_dir = '/test/HR'

# Single SR output from LR image config
run_single_output = True
downsample_input_image = True
image_filename = '0a4baa7dea1785.jpg'   # change this to your Low Resolution input image

# Single Output Paths
input_hr_dir = 'single_output/input/HR'
input_lr_dir = 'single_output/input/LR'
single_output_dir = 'single_output/output/HR'

# ======= Config Parameters =======

# ============ Training Functions ============

# Training loop
def train_srrescycgan(generator_sr, generator_lr, discriminator_hr, discriminator_lr, data_loader, optimizer_g, optimizer_d, losses, num_epochs, changing_LR=False, save_model_weights_per_epoch=False):
    # Initialize scaler
    scaler = GradScaler()

    g_history = [0] * num_epochs
    d_history = [0] * num_epochs

    # scheduler_g is for generator, scheduler_d is for discriminator.
    if (changing_LR):
        scheduler_g = StepLR(optimizer_g, step_size=5, gamma=0.5)
        scheduler_d = StepLR(optimizer_g, step_size=5, gamma=0.5)
    for epoch in range(num_epochs):
        count = 0
        for i, (lr, hr) in enumerate(data_loader):
            lr, hr = lr.to(device), hr.to(device)
            count = count + 1
            if (count % 50 == 0):  # Print the number of batches passed every 50 batches.
                print('Batch Number:', count)

            # Train Generators
            optimizer_g.zero_grad()
            with autocast():
                sr = generator_sr(lr)
                lr_recon = generator_lr(sr)
                sr_pred = discriminator_hr(sr)
                hr_pred = discriminator_hr(hr)

                loss_g = losses.total_loss(sr, hr, lr_recon, lr, sr_pred, hr_pred)

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            # Train Discriminators
            optimizer_d.zero_grad()
            with autocast():
                sr_pred = discriminator_hr(sr.detach())
                hr_pred = discriminator_hr(hr)
                loss_d_hr = losses.gan_loss(sr_pred, hr_pred)

                lr_pred = discriminator_lr(lr_recon.detach())
                lr_real_pred = discriminator_lr(lr)

                loss_d_lr = losses.gan_loss(lr_pred, lr_real_pred)

                loss_d = loss_d_hr + loss_d_lr

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss G: {loss_g.item()}, Loss D: {loss_d.item()}")
        if save_model_weights_per_epoch:
            save_weights(generator_sr, generator_lr, discriminator_hr, discriminator_lr,epoch)
        g_history[epoch] = loss_g.item()  # Save generator loss for later plotting.
        d_history[epoch] = loss_d.item()  # Save discriminator loss for later plotting.

        if changing_LR:
            scheduler_g.step()
            scheduler_d.step()

    return g_history, d_history


# Training loop only for generator
def train_generator_only(generator_sr, generator_lr, discriminator_hr, data_loader, optimizer_g, losses, num_epochs, changing_LR=False, save_model_weights_per_epoch=False):
    scaler = GradScaler()
    g_history = [0] * num_epochs

    if changing_LR:
        scheduler_g = StepLR(optimizer_g, step_size=10,
                             gamma=0.5)  # To use changing learning rate uncomment this line and set them according to your requirements.

    for epoch in range(num_epochs):
        count = 0
        for i, (lr, hr) in enumerate(data_loader):
            lr, hr = lr.to(device), hr.to(device)
            count = count + 1
            if (count % 50 == 0):  # Print the batch number every 50 batches.
                print('Batch Number:', count)

            # Train Generator
            optimizer_g.zero_grad()
            with autocast():
                sr = generator_sr(lr)
                lr_recon = generator_lr(sr)
                sr_pred = discriminator_hr(sr)
                hr_pred = discriminator_hr(hr)

                loss_g = losses.total_loss(sr, hr, lr_recon, lr, sr_pred, hr_pred)

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss G: {loss_g.item()}")
        if save_model_weights_per_epoch:
            save_generator_weights(generator_sr, generator_lr, epoch)
        g_history[epoch] = loss_g.item()  # Save the generator's loss for later plotting.

        if changing_LR:
            scheduler_g.step()  # Uncomment this if you are using changing LR.

    return g_history


def freeze_layers(model, num_layers_to_train=2):
    layers = list(model.children())
    num_layers = len(layers)

    # Freeze all layers except the last num_layers_to_train layers
    for i, layer in enumerate(layers[:-num_layers_to_train]):
        for param in layer.parameters():
            param.requires_grad = False

# ============ Training Functions ============


# ============ Run Training ============

if run_training:
    print("Training Enabled. Running train...")

    # Define a transform to resize images to 256x256 and convert to tensor
    transform = Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    dataset = SRDataset(lr_dir, hr_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=3, num_workers=4, shuffle=True, pin_memory=True)  # Build Batches

    # Print chapes of each low resolution and high resolution image (optional)
    for lr, hr in data_loader:
        print(lr.shape, hr.shape)

    # Model, Optimizer, and Losses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator_sr = GSR().to(device)
    generator_lr = GLR().to(device)
    discriminator_hr = Discriminator().to(device)
    discriminator_lr = Discriminator().to(device)
    vgg = VGG19().to(device)
    optimizer_g = torch.optim.Adam(list(generator_sr.parameters()) + list(generator_lr.parameters()), lr=0.00001,
                               betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    optimizer_d = torch.optim.Adam(list(discriminator_hr.parameters()) + list(discriminator_lr.parameters()), lr=0.00001,
                               betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    losses = SRResCycGANLosses(vgg=vgg)

    if load_nets:
        load_pretrained_nets(generator_sr, generator_lr, discriminator_hr, discriminator_lr, optimizer_g, optimizer_d)

    # Freeze all layers except the last two in the generator
    if freeze_fine_tune_layers:
        freeze_layers(generator_sr, num_layers_to_train=2)
        freeze_layers(generator_lr, num_layers_to_train=2)
        optimizer_g = torch.optim.Adam(filter(lambda p: p.requires_grad, generator_sr.parameters()), lr=0.00001)

    if only_train_generator:
        g_history = train_generator_only(generator_sr, generator_lr, discriminator_hr, data_loader, optimizer_g, losses, num_epochs=30, changing_LR=changing_LR, save_model_weights_per_epoch=save_model_weights_per_epoch)
        if plot_loss_graphs:
            plot_losses('Generator', g_history)
    else:
        g_history, d_history = train_srrescycgan(generator_sr, generator_lr, discriminator_hr, discriminator_lr, data_loader, optimizer_g, optimizer_d, losses, num_epochs=30, changing_LR=changing_LR, save_model_weights_per_epoch=save_model_weights_per_epoch)
        if plot_loss_graphs:
            plot_losses('Generator', g_history)
            plot_losses('Discriminator', d_history)
            plot_losses_vs(g_history, d_history)

    print("Training completed successfully!")

# ============ Run Training ============


# ============ Run Test ============

if run_test:
    print("Test Enabled. Running test...")

    # Initialize LPIPS model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # Define a transform to resize images to 256x256 and convert to tensor
    transform = Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    dataset = SRDataset(lr_test_dir, hr_test_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluate models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator_sr = GSR().to(device)

    avg_ssim_pe = [0] * num_models
    avg_psnr_pe = [0] * num_models
    avg_lpips_pe = [0] * num_models

    for i in range(num_models):
        generator_sr.load_state_dict(torch.load(f'models/generator_sr_epoch_{i + 1}.pth')) # Load the weights of the given model (per epoch).
        generator_sr.eval()
        avg_ssim, avg_psnr, avg_lpips = evaluate_model(dataloader, generator_sr, loss_fn_vgg, device)

        print(f"Average SSIM {i + 1}: {avg_ssim}")
        print(f"Average PSNR {i + 1}: {avg_psnr} dB")
        print(f"Average LPIPS {i + 1}: {avg_lpips}")
        print('')

    avg_ssim_pe[i] = avg_ssim
    avg_psnr_pe[i] = avg_psnr
    avg_lpips_pe[i] = avg_lpips

    if plot_metrics_graphs:
        plot_metric('SSIM', avg_ssim_pe)
        plot_metric('PSNR', avg_psnr_pe)
        plot_metric('LPIPS', avg_lpips_pe)

    print("Test completed successfully!")
# ============ Run Test ============

# ============ Get Output SR Image ============

if run_single_output:
    print("Single output enabled! Generating single output SR image from LR input...")

    # Model, Optimizer, and Losses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator_sr = GSR().to(device)

    # Load the pre-trained weights
    generator_sr.load_state_dict(torch.load('models/generator_sr.pth'))

    # Set the model to evaluation mode
    generator_sr.eval()

    if downsample_input_image:
        downsample_hr_dataset(input_hr_dir, input_lr_dir)

    # Load the low-resolution image
    lr_image = Image.open(f"{input_lr_dir}/{image_filename}").convert('RGB')

    # Preprocess the image: Convert to tensor and add a batch dimension
    transform = ToTensor()
    lr_image_tensor = transform(lr_image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Generate the high-resolution image
    with torch.no_grad():  # Disable gradient calculation for inference
        output = generator_sr(lr_image_tensor)

    # Remove the batch dimension and convert back to a PIL image
    sr_image_tensor = output.squeeze(0)  # Remove batch dimension
    to_pil = ToPILImage()
    sr_image = to_pil(sr_image_tensor.cpu())  # Move to CPU and convert to PIL image

    # Save the high-resolution image
    sr_image.save(f'{single_output_dir}/super_resolved_image.jpg')

    print('Image generated successfully! Please check the output directory to see the generated image.')
