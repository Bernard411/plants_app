from django.shortcuts import render
from .models import Disease, FAQGroup, FAQItem
# Create your views here.

def home(request):
    plants = Disease.objects.all()
    faq_groups = FAQGroup.objects.prefetch_related('faq_items').all()
    context ={
        'plants' : plants,
        'faq_groups': faq_groups
    }
    return render(request, 'index.html', context)



from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.utils.datastructures import MultiValueDictKeyError

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.utils.datastructures import MultiValueDictKeyError
from .dl_model.model import classify_image



from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import PlantDiseaseForm_x
from .dl_model.model import classify_image  # Adjust the import based on your project structure

def scan(request):
    if request.method == 'POST':
        form = PlantDiseaseForm_x(request.POST, request.FILES)
        if form.is_valid():
            description = form.cleaned_data['description']
            image = form.cleaned_data['image']
            
            # Convert the image file to bytes
            image_bytes = image.read()
            
            # Predict the class of the image
            result = classify_image(image_bytes)
            
            # Select the top three predictions according to their probabilities
            top1 = '1. Species: %s, Status: %s, Probability: %.4f' % (result[0][0], result[0][1], result[0][2])
            top2 = '2. Species: %s, Status: %s, Probability: %.4f' % (result[1][0], result[1][1], result[1][2])
            top3 = '3. Species: %s, Status: %s, Probability: %.4f' % (result[2][0], result[2][1], result[2][2])

            predictions = [{'pred': top1}, {'pred': top2}, {'pred': top3}]
            context = {'predictions': predictions, 'description': description}
            
            # Save the file to ./media
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_file_url = fs.url(filename)
            context['url'] = uploaded_file_url

            return render(request, 'scan.html', context)
    else:
        form = PlantDiseaseForm_x()

    return render(request, 'scan.html', {'form': form})







from django.shortcuts import render
from .forms import DiagnosisForm

def using_form(request):
    if request.method == 'POST':
        form = DiagnosisForm(request.POST)
        if form.is_valid():
            # Extract form data
            data = form.cleaned_data
            
            # Implement your algorithm to predict the disease
            diagnosis = predict_disease(data)
            
            return render(request, 'results.html', {'diagnosis': diagnosis})
    else:
        form = DiagnosisForm()
    
    return render(request, 'forms.html', {'form': form})

# disease_checker/views.py

from django.shortcuts import render
from .utils import predict_disease
from .forms import DiseasePredictionForm  # Create this form as per your requirements

def using_form(request):
    if request.method == 'POST':
        form = DiseasePredictionForm(request.POST)
        if form.is_valid():
            data = {
                'leaf_color': form.cleaned_data['leaf_color'],
                'leaf_shape': form.cleaned_data['leaf_shape'],
                'spots_on_leaves': form.cleaned_data['spots_on_leaves'],
                'wilting': form.cleaned_data['wilting'],
                'mold_growth': form.cleaned_data['mold_growth'],
                'stem_condition': form.cleaned_data['stem_condition'],
                'soil_type': form.cleaned_data['soil_type'],
                'weather_conditions': form.cleaned_data['weather_conditions'],
                'watering_frequency': form.cleaned_data['watering_frequency'],
                'pest_presence': form.cleaned_data['pest_presence'],
                'fertilizer_use': form.cleaned_data['fertilizer_use'],
                'crop_type': form.cleaned_data['crop_type'],
            }
            diagnosis = predict_disease(data)
            faq_groups = FAQGroup.objects.prefetch_related('faq_items').all()
           
            return render(request, 'results.html', {'diagnosis': diagnosis, 'faq_groups': faq_groups})
    else:
        form = DiseasePredictionForm()
    
    return render(request, 'forms.html', {'form': form})


def results(request):
    faq_groups = FAQGroup.objects.prefetch_related('faq_items').all()
    context ={
       
        'faq_groups': faq_groups
    }
    return render(request, 'results.html', context)










from django.contrib.auth import logout
from django.shortcuts import redirect
from .models import *
from django.shortcuts import render
from django.contrib.auth import authenticate, login
from django.contrib import messages


def logout_view(request):
    logout(request)
    # Redirect to a success page or home page.
    return redirect('login')  # Replace 'home' with the name of your home URL pattern.

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username and password:  # Check if both username and password are provided
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
            
                return redirect('homepage')  # Assuming you have a 'success' URL name defined in your urls.py
            else:
               
                messages.error(request, 'Invalid username or password.')
        else:
            # Return an error message if either username or password is missing
            messages.error(request, 'Please provide both username and password.')
    return render(request, 'login.html')  # Assuming you have a login.html template in your templates directory

from .forms import RegistrationForm

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            
            # Authenticate the user
            new_user = authenticate(username=form.cleaned_data['username'],
                                    password=form.cleaned_data['password1'])
            if new_user is not None:
                login(request, new_user)  # Log in the user
                return redirect('login')  # Redirect to the home page after successful registration
            
    else:
        form = RegistrationForm()
    return render(request, 'register.html', {'form': form})

def view_plant(request, pk):
    plants = Disease.objects.get(pk=pk)
    context = {
        'plants' : plants
    }
    return render(request, 'view.html', context)