from django.db import models

# Create your models here.

class User(models.Model):
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=10, null=False)
    password = models.CharField(max_length=20, null=False)
    is_admin = models.BooleanField(default=False)
    email = models.CharField(max_length=30, unique=True)
    institution = models.CharField(max_length=20)
    age = models.IntegerField(default=18)
    gender = models.IntegerField(default=0)
    reason = models.IntegerField(default=0)

class Net(models.Model):
    id = models.AutoField(primary_key=True)
    net_name = models.CharField(max_length=15, null=False)
    net_file = models.CharField(max_length=20, null=False)
    description = models.TextField()
    in_channel = models.IntegerField()
    in_size = models.IntegerField()

class Record(models.Model):
    id = models.AutoField(primary_key=True)
    input_url = models.CharField(max_length=100)
    upload_by = models.ForeignKey(User, on_delete=models.CASCADE)
    upload_time = models.DateTimeField(auto_now_add=True)
    chosen_net = models.CharField(max_length=15)
    output_url = models.CharField(max_length=100)


