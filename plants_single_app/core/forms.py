from django import forms

# forms.py
from django import forms

class DiagnosisForm(forms.Form):
    LEAF_COLOR_CHOICES = [
        ('green', 'Green'),
        ('yellow', 'Yellow'),
        ('brown', 'Brown'),
        ('black', 'Black'),
        ('purple', 'Purple'),
    ]
    
    LEAF_SHAPE_CHOICES = [
        ('oval', 'Oval'),
        ('heart', 'Heart'),
        ('lance', 'Lance'),
        ('round', 'Round'),
    ]
    
    STEM_CONDITION_CHOICES = [
        ('healthy', 'Healthy'),
        ('soft', 'Soft'),
        ('brittle', 'Brittle'),
        ('discolored', 'Discolored'),
    ]
    
    SOIL_TYPE_CHOICES = [
        ('clay', 'Clay'),
        ('sandy', 'Sandy'),
        ('silty', 'Silty'),
        ('peaty', 'Peaty'),
        ('chalky', 'Chalky'),
        ('loamy', 'Loamy'),
    ]
    
    WEATHER_CONDITIONS_CHOICES = [
        ('sunny', 'Sunny'),
        ('cloudy', 'Cloudy'),
        ('rainy', 'Rainy'),
        ('windy', 'Windy'),
    ]
    
    WATERING_FREQUENCY_CHOICES = [
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('biweekly', 'Biweekly'),
        ('monthly', 'Monthly'),
    ]
    
    leaf_color = forms.ChoiceField(choices=LEAF_COLOR_CHOICES)
    leaf_shape = forms.ChoiceField(choices=LEAF_SHAPE_CHOICES)
    spots_on_leaves = forms.BooleanField(required=False)
    wilting = forms.BooleanField(required=False)
    mold_growth = forms.BooleanField(required=False)
    stem_condition = forms.ChoiceField(choices=STEM_CONDITION_CHOICES)
    soil_type = forms.ChoiceField(choices=SOIL_TYPE_CHOICES)
    weather_conditions = forms.ChoiceField(choices=WEATHER_CONDITIONS_CHOICES)
    watering_frequency = forms.ChoiceField(choices=WATERING_FREQUENCY_CHOICES)
    pest_presence = forms.BooleanField(required=False)
    fertilizer_use = forms.CharField(max_length=100, required=False)
    crop_type = forms.CharField(max_length=100)


from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class RegistrationForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super(RegistrationForm, self).save(commit=False)
        user.email = self.cleaned_data['username']
        if commit:
            user.save()
            # Automatically create a profile for the user
            
        return user
    

# disease_checker/forms.py

from django import forms

class DiseasePredictionForm(forms.Form):
    LEAF_COLORS = [
        ('yellow', 'Yellow'),
        ('brown', 'Brown'),
        ('black', 'Black'),
        ('purple', 'Purple'),
    ]

    LEAF_SHAPES = [
        ('oval', 'Oval'),
        ('heart', 'Heart'),
        ('lance', 'Lance'),
    ]

    STEM_CONDITIONS = [
        ('healthy', 'Healthy'),
        ('discolored', 'Discolored'),
        ('brittle', 'Brittle'),
        ('soft', 'Soft'),
    ]

    SOIL_TYPES = [
        ('sandy', 'Sandy'),
        ('clay', 'Clay'),
        ('silty', 'Silty'),
    ]

    WEATHER_CONDITIONS = [
        ('sunny', 'Sunny'),
        ('rainy', 'Rainy'),
        ('windy', 'Windy'),
    ]

    WATERING_FREQUENCIES = [
        ('weekly', 'Weekly'),
        ('biweekly', 'Biweekly'),
        ('monthly', 'Monthly'),
    ]

    FERTILIZER_USE = [
        ('none', 'None'),
        ('organic', 'Organic'),
        ('chemical', 'Chemical'),
    ]

    leaf_color = forms.ChoiceField(choices=LEAF_COLORS, label='Leaf Color')
    leaf_shape = forms.ChoiceField(choices=LEAF_SHAPES, label='Leaf Shape')
    spots_on_leaves = forms.BooleanField(label='Spots on Leaves', required=False)
    wilting = forms.BooleanField(label='Wilting', required=False)
    mold_growth = forms.BooleanField(label='Mold Growth', required=False)
    stem_condition = forms.ChoiceField(choices=STEM_CONDITIONS, label='Stem Condition')
    soil_type = forms.ChoiceField(choices=SOIL_TYPES, label='Soil Type')
    weather_conditions = forms.ChoiceField(choices=WEATHER_CONDITIONS, label='Weather Conditions')
    watering_frequency = forms.ChoiceField(choices=WATERING_FREQUENCIES, label='Watering Frequency')
    pest_presence = forms.BooleanField(label='Pest Presence', required=False)
    fertilizer_use = forms.ChoiceField(choices=FERTILIZER_USE, label='Fertilizer Use')
    crop_type = forms.CharField(max_length=100, label='Crop Type')

class PlantDiseaseForm_x(forms.Form):
    description = forms.CharField(widget=forms.Textarea, required=True, label='Describe the issue with the plant')
    image = forms.ImageField(required=True, label='Upload an image of the plant')