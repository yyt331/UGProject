from django.urls import path
from . import views  
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('upload/', views.upload_image, name='upload_image'),
    path('delete/<int:pk>/', views.delete_image, name='delete_image'),
    path('results/<int:pk>/', views.display_results, name='display_results'), 
]
