from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import Images, SimilarImages
from .forms import ImageUploadForm
from .utils import classify_image, get_embedding, find_similar_images


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            new_image = form.save()
            image_bytes = new_image.image.read()

            # Immediately process the image
            label, confidence = classify_image(image_bytes)
            embedding = get_embedding(image_bytes)
            similar_image_paths = find_similar_images(embedding)

            # Store results
            new_image.prediction = label
            new_image.confidence = confidence
            new_image.save()

            # Create and store relationships for similar images
            for path in similar_image_paths:
                similar_img, created = Images.objects.get_or_create(image=path)
                SimilarImages.objects.create(original_image=new_image, similar_image=similar_img)

            # Redirect to the results page with the ID of the new image
            return redirect('display_results', pk=new_image.pk)

    else:
        form = ImageUploadForm()
    
    return render(request, 'upload.html', {'form': form})

def delete_image(request, pk):
    # Get the image object, delete it and redirect to the upload page
    image = get_object_or_404(Images, pk=pk)
    image.delete()
    return redirect('upload_image')

def display_results(request, pk):
    # Get the original image and its related similar images
    image_instance = get_object_or_404(Images, pk=pk)
    image_embedding = get_embedding(image_instance.image.read())

    similar_image_urls = find_similar_images(image_embedding)

    print(similar_image_urls) 

    # Render the results
    return render(request, 'results.html', {
        'image': image_instance,
        'similar_images': similar_image_urls
    })

