from django.urls import path
from . import views

app_name = 'Locator'

urlpatterns = [
    path('', views.ConstellationFinderView.as_view(), name='locator'),
    path('save-location/', views.save_location, name='save_location'),
    path('find-constellations/', views.find_constellations, name='find_constellations'),
]