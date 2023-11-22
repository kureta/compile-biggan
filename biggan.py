import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample, save_as_images)
import PIL.Image  # noqa

MODEL = 'biggan-deep-512'


def main():
    # Load pre-trained model tokenizer (vocabulary)
    model = BigGAN.from_pretrained(MODEL)

    # Prepare a input
    truncation = 0.4
    class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=3)
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3)

    # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    # If you have a GPU, put everything on cuda
    noise_vector = noise_vector.to('cuda')
    class_vector = class_vector.to('cuda')
    model.to('cuda')

    # Generate an image
    with torch.no_grad():
        output = model(noise_vector, class_vector, torch.tensor(truncation))

    # If you have a GPU put back on CPU
    output = output.to('cpu')

    # Save results as png images
    save_as_images(output)

    # Save model as torch-script
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    script_module = torch.jit.trace(model, (noise_vector, class_vector, torch.tensor(truncation)))
    script_module.save(f'{MODEL}.pt')


if __name__ == '__main__':
    main()
