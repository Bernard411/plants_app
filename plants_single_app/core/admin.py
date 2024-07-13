from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register(Disease)
admin.site.register(FAQGroup)
admin.site.register(FAQItem)