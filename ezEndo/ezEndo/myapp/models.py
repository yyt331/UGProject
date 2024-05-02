from django.db import models

# Create your models here.

class Images(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    prediction = models.CharField(max_length=10, blank=True)
    confidence = models.FloatField(default=0.00, blank=True)

    def __str__(self):
        return f"{self.prediction} ({self.confidence}%)"
    
class SimilarImages(models.Model):
    original_image = models.ForeignKey(Images, related_name='similarity_instances', on_delete=models.CASCADE)
    similar_image = models.ForeignKey(Images, related_name='similar_to', on_delete=models.CASCADE)

    def __str__(self):
        return f"Original Image: {self.original_image.id}, Similar Image: {self.similar_image.id}"