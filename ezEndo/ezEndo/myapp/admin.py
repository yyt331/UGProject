from django.contrib import admin
from .models import Images, SimilarImages

# Register your models here.
@admin.register(Images)
class ImagesAdmin(admin.ModelAdmin):
    list = ('id', 'prediction', 'confidence')

@admin.register(SimilarImages)
class SimilarImagesAdmin(admin.ModelAdmin):
    list = ('orginial_image', 'similar_images')