# diagnosis/models.py
from django.db import models

class Disease(models.Model):
    name = models.CharField(max_length=100)
    symptoms = models.TextField()
    description = models.TextField()
    prevention = models.TextField()
    treatment = models.TextField()
    image = models.URLField()

    def __str__(self):
        return self.name


class FAQGroup(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class FAQItem(models.Model):
    group = models.ForeignKey(FAQGroup, related_name='faq_items', on_delete=models.CASCADE)
    question = models.CharField(max_length=200)
    answer = models.TextField()

    def __str__(self):
        return self.question
