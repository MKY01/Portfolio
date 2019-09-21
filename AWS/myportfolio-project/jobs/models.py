from django.db import models

class Job(models.Model):
    #https://docs.djangoproject.com/en/2.2/ref/models/fields/
    image = models.ImageField(upload_to='images/')
    summary = models.CharField(max_length=200)
