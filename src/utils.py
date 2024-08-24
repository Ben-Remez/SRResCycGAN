import os
import cv2
import torch
import shutil
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.metrics import structural_similarity as SSIM

# Function to filter images by resolution
def filter_images_by_resolution(image_files, min_width=2048, min_height=2048, output_path='/data/filtered_images/HR'):
    filtered_images = []

    for image_file in image_files:
        with Image.open(image_file) as img:
            width, height = img.size
            if width >= min_width or height >= min_height:
                if img.mode == 'RGBA':  # Check if the image is in RGBA mode
                    img = img.convert('RGB')  # Convert to RGB mode

                filtered_images.append(image_file)

                # Save the filtered images to a new directory
                img.save(os.path.join(output_path, os.path.basename(image_file)))

    return filtered_images


def filter_and_save_images_by_res(dataset_path, output_path):
    # Path to your dataset Example path: '/content/drive/My Drive/example_set'
    dataset_path = dataset_path

    # Directory where you want to save the filtered images Example Path: '/content/drive/My Drive/filtered_set/HR'
    output_path = output_path
    os.makedirs(output_path, exist_ok=True)

    # List all image files in the directory
    image_files = [os.path.join(dataset_path, fname) for fname in os.listdir(dataset_path) if
                   fname.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    filtered_images = filter_images_by_resolution(image_files, output_path=output_path)

    print(f"Found {len(filtered_images)} images with a width or height of 2K or higher.")


# Funtion that creates a dataset from the top number of images from a given dataset.
def generate_top_hr_imgs_dataset(dataset_path='/data/HR', output_dir='/data/top_images/HR', num_of_images=1000):
    # List all image files in the directory
    image_files = [os.path.join(dataset_path, fname) for fname in os.listdir(dataset_path) if
                   fname.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    # Create a list to hold image paths and their resolutions
    image_resolutions = []

    # Iterate over the images and get their resolutions
    for image_file in image_files:
        with Image.open(image_file) as img:
            width, height = img.size
            resolution = width * height
            image_resolutions.append((image_file, resolution))

    # Sort images by resolution in descending order
    sorted_images = sorted(image_resolutions, key=lambda x: x[1], reverse=True)

    # Select the top number of images accodring to num_of_images variable
    top_images = sorted_images[:num_of_images]

    # Extract the file paths of top images
    top_image_paths = [image[0] for image in top_images]

    # Print the selected top images (optional)
    for img_path in top_image_paths:
        print(img_path)

    os.makedirs(output_dir, exist_ok=True)

    for img_path in top_image_paths:
        shutil.copy(img_path, output_dir)


def get_y_axis_rotation_matrix(w, h, angle):
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Calculate the new width after rotation
    new_w = w * cos_angle

    # Calculate the destination points for perspective transform
    src_points = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    dst_points = np.float32([
        [(w - new_w) / 2, 0],
        [(w + new_w) / 2, 0],
        [(w + new_w) / 2, h],
        [(w - new_w) / 2, h]
    ])

    return cv2.getPerspectiveTransform(src_points, dst_points), (w, h)


def apply_y_axis_rotation(image, angle):
    h, w = image.shape[:2]
    M, new_size = get_y_axis_rotation_matrix(w, h, angle)
    transformed_image = cv2.warpPerspective(image, M, new_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return transformed_image


def process_directory(input_dir, output_dir, num_angles=5, angle_range=(80, 89)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            base_filename = os.path.splitext(filename)[0]

            for i in range(num_angles):
                angle = random.uniform(*angle_range)
                transformed_image = apply_y_axis_rotation(image, angle)
                output_filename = f"{base_filename}_rotated_{i+1}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, transformed_image)


def rotate_images(input_directory='/data/HR', output_directory='/data/rotated/HR'):
    # Change the angle according to your desired rotation.
    process_directory(input_directory, output_directory, num_angles=1, angle_range=(40, 70))


def downsample_image(image_path, output_path, scale_factor=4):
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA' or img.mode == 'P':
            img = img.convert('RGB')
        lr_img = img.resize((img.width // scale_factor, img.height // scale_factor), Image.BICUBIC)
        lr_img.save(output_path, 'JPEG')


# Downsample HR images to create LR images
def downsample_hr_dataset(hr_folder='/data/HR', lr_folder='/data/LR'):
    os.makedirs(lr_folder, exist_ok=True)
    count = 1
    for filename in os.listdir(hr_folder):
        if count % 50 == 0:  # Print the number of batch and image file name every 50 images.
            print(count, filename)
        count += 1
        if filename.endswith(".png") or filename.endswith(".jpg"):
            hr_image_path = os.path.join(hr_folder, filename)
            lr_image_path = os.path.join(lr_folder, filename)
            downsample_image(hr_image_path, lr_image_path, scale_factor=4)  # Change the scale factor as you wish.


# A function to save the weights for a desired epoch.
def save_weights(generator_sr, generator_lr, discriminator_sr, discriminator_lr, epoch):
    save_num = epoch + 1
    sr_g_file_name = f"/weights/generator_sr_epoch_{save_num}.pth"
    lr_g_file_name = f"/weights/generator_lr_epoch_{save_num}.pth"
    sr_d_file_name = f"/weights/discriminator_sr_epoch_{save_num}.pth"
    lr_d_file_name = f"/weights/discriminator_lr_epoch_{save_num}.pth"
    torch.save(generator_sr.state_dict(), sr_g_file_name)
    torch.save(generator_lr.state_dict(), lr_g_file_name)
    torch.save(discriminator_sr.state_dict(), sr_d_file_name)
    torch.save(discriminator_lr.state_dict(), lr_d_file_name)


# A function to save only generator weights for a desired epoch. (Used when only training generator)
def save_generator_weights(generator_sr, generator_lr, epoch):
    save_num = epoch + 1
    sr_g_file_name = f"/weights/generator_sr_epoch_{save_num}.pth"
    lr_g_file_name = f"/weights/generator_lr_epoch_{save_num}.pth"
    torch.save(generator_sr.state_dict(), sr_g_file_name)
    torch.save(generator_lr.state_dict(), lr_g_file_name)


# If you decided to remove the save for each epoch in the training loop, this can be used to save all the model's components weights.
def save_model_weights(generator_sr, generator_lr, discriminator_hr, discriminator_lr, optimizer_g, optimizer_d):
    # Save the generator and discriminator weights
    torch.save(generator_sr.state_dict(), 'data/weights/generator_sr.pth')
    torch.save(generator_lr.state_dict(), 'data/weights/generator_lr.pth')
    torch.save(discriminator_hr.state_dict(), 'data/weights/discriminator_hr.pth')
    torch.save(discriminator_lr.state_dict(), 'data/weights/discriminator_lr.pth')

    # Optionally, save the optimizer states as well
    torch.save(optimizer_g.state_dict(), 'data/weights/optimizer_g.pth')
    torch.save(optimizer_d.state_dict(), 'data/weights/optimizer_d.pth')


def load_pretrained_nets(generator_sr, generator_lr, discriminator_hr, discriminator_lr, optimizer_g, optimizer_d):
    # If the training starts from scratch and no weights are loaded, comment these lines.
    generator_sr.load_state_dict(torch.load('/pretrained_nets/generator_sr.pth'))
    generator_lr.load_state_dict(torch.load('/pretrained_nets/generator_lr.pth'))

    discriminator_hr.load_state_dict(torch.load('/pretrained_nets/discriminator_hr.pth'))
    discriminator_lr.load_state_dict(torch.load('/pretrained_nets/discriminator_lr.pth'))

    optimizer_g.load_state_dict(torch.load('/pretrained_nets/optimizer_g.pth'))
    optimizer_d.load_state_dict(torch.load('/pretrained_nets/optimizer_d.pth'))


# A function to plot the loss graph for discriminator/generator
def plot_losses(g_or_d, history):
    epochs = range(1, len(history) + 1)

    # Plot the loss vs. epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history, 'b-', marker='o', label=f'Loss {g_or_d}')  # 'b-' means blue line
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss vs. Epoch {g_or_d}')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(history) + 1))
    plt.savefig(f'loss_vs_epoch_{g_or_d}.png')
    plt.show()


# A function to plot both discriminator and generator losses in a single graph
def plot_losses_vs(g_history, d_history):
    epochs = range(1, len(g_history) + 1)

    # Plot the loss vs. epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, g_history, 'b-', marker='o', label=f'Loss Generator')
    plt.plot(epochs, d_history, 'r-', marker='o', label=f'Loss Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(g_history) + 1))
    plt.savefig('loss_vs_epoch.png')
    plt.show()


def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, win_size=7):
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and NumPy array
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and NumPy array
    ssim_value = SSIM(img1_np, img2_np, multichannel=True, data_range=img1_np.max() - img1_np.min(), win_size=win_size)
    return ssim_value


def calculate_lpips(img1, img2, loss_fn):
    # Ensure img1 and img2 are normalized and have the same dimensions
    lpips_value = loss_fn(img1, img2)
    return lpips_value.item()


# Function to compute metrics
def compute_metrics(lr_image, hr_image, sr_image, loss_fn_vgg):
    # Resize SR image tensor to match the dimensions of HR image tensor
    sr_image_resized = F.interpolate(sr_image, size=hr_image.shape[2:], mode='bilinear', align_corners=False)

    # Calculate metrics using the resized SR image
    psnr_value = calculate_psnr(sr_image_resized, hr_image)
    ssim_value = calculate_ssim(sr_image_resized, hr_image, win_size=3)
    lpips_value = calculate_lpips(sr_image_resized, hr_image, loss_fn_vgg)

    return ssim_value, psnr_value, lpips_value


# Function to evaluate the model on the entire test set
def evaluate_model(dataloader, model, loss_fn_vgg, device):
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []

    for lr_image, hr_image in dataloader:
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)

        # Generate the super-resolved image
        with torch.no_grad():
            sr_image = model(lr_image)

        # Compute metrics
        ssim_value, psnr_value, lpips_value = compute_metrics(lr_image, hr_image, sr_image, loss_fn_vgg)

        # Store the metrics
        ssim_scores.append(ssim_value)
        psnr_scores.append(psnr_value)
        lpips_scores.append(lpips_value)

    # Calculate the average metrics
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_lpips = np.mean(lpips_scores)

    return avg_ssim, avg_psnr, avg_lpips


def plot_metric(metric_name, history):
    epochs = range(1, len(history) + 1)

    # Plot the metric vs. epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history, 'b-', marker='o', label=metric_name)
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_name}')
    plt.title(f'{metric_name} vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(history) + 1))
    plt.savefig(f'{metric_name}_vs_epoch.png')
    plt.show()

