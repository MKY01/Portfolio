from django.shortcuts import render
from .forms import HashForm

def home(request):
    form = HashForm()
    return render(request, 'hashing/home.html', {'form':form})
