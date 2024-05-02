from django import forms
from .models import Images
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Images
        fields = ['image']

    def clean_image(self):
        image = self.cleaned_data.get('image', False)
        if image:
            if not image.content_type in ['image/jpeg', 'image/png', 'image/gif']:
                raise ValidationError(_("File is not JPEG, PNG, or GIF"))
            return image
        else:
            raise ValidationError(_("Couldn't read uploaded image"))