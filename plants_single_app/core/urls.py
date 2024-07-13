from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('wellcome/', views.home, name="homepage"),
    path('scan/', views.scan, name="scan"),
    path('resul/', views.results, name="results"),
    path('using/forms/', views.using_form, name="entity"),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('', views.login_view, name='login'),
    path('plant/d/<int:pk>', views.view_plant, name="view_plant")
]