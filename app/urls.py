from django.urls import path

from app.views import Index, Login, Nets, Register, Predict, Admin, deleteNet, deleteRecord, downloadPic, Process

app_name = 'app'


urlpatterns = [
    path('', Index.as_view(), name='index'),
    path('login', Login, name='login'),
    path('nets', Nets.as_view(), name='nets'),
    path('process', Process, name='process'),
    path('register', Register, name='register'),
    path('predict/<int:net_id>', Predict, name='predict'),
    path('admin/<int:page>', Admin, name='admin'),
    path('deleteNet/<int:net_id>/', deleteNet, name='deleteNet'),
    path('deleteRecord/<int:record_id>/', deleteRecord, name='deleteRecord'),
    path('downloadPic/<str:out_url>', downloadPic, name='downloadPic')
]