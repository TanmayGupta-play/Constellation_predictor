from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot, name='chatbot'),
    path('api/ask/', views.ask_ai, name='ask_ai'),
    path('speech_to_text/', views.speech_to_text, name='speech_to_text'),  
    path('text_to_speech/', views.text_to_speech, name='text_to_speech'),]