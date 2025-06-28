from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detect/', views.detect_view, name='detect'),
path('detect/process/', views.process_frame, name='process_frame'),
path('chatbot/', include('chatbot.urls')),
path('locator/', include('Locator.urls'),name='Locator'),
path('database/', views.database, name='database'),
    path('predict/', views.predict, name='predict'),
    path('upload/',views.upload,name='upload'),
    path('get-constellation-info/', views.get_constellation_info, name='get_constellation_info'),
    path('process-upload/', views.process_upload, name='process_upload'),
    
]
