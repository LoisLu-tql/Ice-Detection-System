import os
import shutil
import sys
import uuid
import zipfile

import h5py
import numpy
import torch
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import render, redirect
import cv2
from django.views.generic import TemplateView
from app import models
from app.models import Net, User, Record
from ices.settings import MEDIA_ROOT, NETS_ROOT

sys.path.append(NETS_ROOT)


class Index(TemplateView):
    template_name = 'index.html'


class Nets(TemplateView):
    template_name = 'nets.html'

    def get_context_data(self, **kwargs):
        nets_list = models.Net.objects.all()
        return {'nets_list': nets_list}


# login
def Login(request):
    error = False
    if request.method == 'POST':
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = models.User.objects.filter(email=email, password=password)
        if not user:
            error = True
        else:
            request.session['user_id'] = user[0].id
            if user[0].is_admin == 1:
                return redirect("app:admin", 1)
            else:
                return redirect("app:nets")
    return render(request, 'login.html', {'error': error})


# register
def Register(request):
    if request.method == 'POST':
        user = User()
        user.username = request.POST.get("username")
        user.password = request.POST.get("password")
        user.email = request.POST.get("email")
        user.institution = request.POST.get("institution")
        user.age = request.POST.get("age")
        user.gender = request.POST.get("gender")
        user.reason = request.POST.get("reason")
        user.save()
        return redirect("app:login")
    else:
        return render(request, 'register.html')


# predict

def ratio_to_label(pred):
    # b, 2, h, w
    mask = numpy.zeros_like(pred[:, 0])
    for b_ind in range(len(pred)):
        mask[b_ind] = pred[b_ind].argmax(axis=0)
    return mask


def downloadPic(request, out_url):
    path = os.path.join(MEDIA_ROOT, out_url)
    with open(path, 'rb') as f:
        file = f.read()
    response = HttpResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    filename = 'attachment; filename=' + '{}'.format(out_url)
    response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
    return response


def Predict(request, net_id):
    # constrain non-login users
    if request.session.get('user_id') is None:
        return redirect('app:login')

    # get corresponding net
    chosen_net = models.Net.objects.get(id=net_id)
    net_name = chosen_net.net_name
    net_file = chosen_net.net_file
    in_channel = chosen_net.in_channel
    in_size = chosen_net.in_size

    if request.method == 'POST':
        files = request.FILES.getlist('imgs')
        if len(files) != in_channel:
            # show error tips
            context = {
                'show': False,
                'chosen_net': chosen_net,
                'error': True
            }
            return render(request, 'predict.html', context)
        out = []
        first = True
        # deal with input imgs
        for file in files:

            if file.name[-4:] != '.png':
                context = {
                    'show': False,
                    'chosen_net': chosen_net,
                    'error': True
                }
                return render(request, 'predict.html', context)

            path = os.path.join(MEDIA_ROOT, file.name)
            with open(path, 'ab') as filepath:
                for chunk in file.chunks():
                    filepath.write(chunk)

            img = cv2.imread(path, 0) / 255.0
            re_img = cv2.resize(img, (in_size, in_size))
            if first:
                in_sum = re_img
                first = False
            else:
                in_sum = in_sum + re_img
            out.append(re_img)
            os.remove(path)

        # fuse the input images
        in_sum = in_sum * 255 / in_channel
        in_filename = str(uuid.uuid4()).replace("-", "") + ".png"
        cv2.imwrite(os.path.join(MEDIA_ROOT, in_filename), in_sum)

        record = Record()
        record.input_url = in_filename
        record.chosen_net = net_name
        record.upload_by = models.User.objects.get(id=request.session.get('user_id'))

        # prepare for predicting
        out = numpy.array([out])
        model_path = os.path.join(NETS_ROOT, net_file)
        model = torch.load(model_path)
        model.eval()
        with torch.no_grad():
            images = out

            pred = model(torch.tensor(images.astype(numpy.float32)))
            pred = ratio_to_label(pred.detach().cpu().numpy()).astype(numpy.uint8)
            output_frames = pred * 255
            cur_img = output_frames[0, :, :]

            out_filename = str(uuid.uuid4()).replace("-", "") + ".png"
            cv2.imwrite(os.path.join(MEDIA_ROOT, out_filename), cur_img)

            record.output_url = out_filename
            record.save()

        context = {
            'out_filename': out_filename,
            'in_filename': in_filename,
            'show': True,
            'chosen_net': chosen_net,
            'error': False
        }
    else:
        context = {
            'show': False,
            'chosen_net': chosen_net,
            'error': False
        }

    return render(request, 'predict.html', context)


# admin management page
def deleteNet(request, net_id):
    models.Net.objects.get(id=net_id).delete()
    return redirect('app:admin', 1)


def deleteRecord(request, record_id):
    os.remove(os.path.join(MEDIA_ROOT, models.Record.objects.get(id=record_id).input_url))
    os.remove(os.path.join(MEDIA_ROOT, models.Record.objects.get(id=record_id).output_url))
    models.Record.objects.get(id=record_id).delete()
    return redirect('app:admin', 1)


def Admin(request, page):
    nets_list = models.Net.objects.all()

    page = page
    limit = 3
    record = models.Record.objects.all().order_by('id')
    paginator = Paginator(record, limit)
    page_now = paginator.get_page(page)

    if request.method == 'POST':
        net = Net()
        net.net_name = request.POST.get("netName")
        net.net_file = request.POST.get("netFile")
        net.description = request.POST.get("netDescription")
        net.in_channel = request.POST.get("inChannel")
        net.in_size = request.POST.get("inSize")
        net.save()
        success = True

    else:
        success = False

    context = {
        'success': success,
        'nets_list': nets_list,
        'page_now': page_now,
        'record': record
    }

    return render(request, 'admin.html', context)


# Process hdf file
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    numpy.seterr(divide="ignore", invalid="ignore")
    return numpy.array((x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)))


def Process(request):
    # constrain non-login users
    if request.session.get('user_id') is None:
        return redirect('app:login')

    if request.method == 'POST':
        file = request.FILES.get("hdf")
        if file.name[-4:] != '.hdf' and file.name[-4:] != '.HDF':
            context = {
                'error': True
            }
            return render(request, 'process.html', context)
        try:
            # read input
            file_dir = os.path.join(MEDIA_ROOT, str(uuid.uuid4()).replace("-", ""))
            os.mkdir(file_dir)
            path = os.path.join(file_dir, file.name)
            with open(path, 'ab') as filepath:
                for chunk in file.chunks():
                    filepath.write(chunk)

            # process
            f = h5py.File(path, 'r')
            data = f['Data']['EV_250_Aggr.1KM_RefSB'][:]
            data = numpy.row_stack((data, f['Data']['EV_1KM_RefSB'][:]))
            data = numpy.row_stack((data, f['Data']['EV_1KM_Emissive'][:]))
            data = numpy.row_stack((data, f['Data']['EV_250_Aggr.1KM_Emissive'][:]))

            zfile = zipfile.ZipFile(os.path.join(MEDIA_ROOT, file.name + ".zip"), "w")

            for i in range(len(data)):
                cv2.imwrite(os.path.join(file_dir, str(i + 1) + ".png"), normalize(data[i]) * 255.0)
                zfile.write(os.path.join(file_dir, str(i + 1) + ".png"))

            zfile.close()

            with open(MEDIA_ROOT + '/' + file.name + ".zip", 'rb') as f:
                zfile = f.read()
            response = HttpResponse(zfile)
            response['Content-Type'] = 'application/octet-stream'
            filename = 'attachment; filename=' + '{}'.format(file.name + ".zip")
            response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')

            shutil.rmtree(file_dir)
            os.remove(MEDIA_ROOT + '/' + file.name + ".zip")

            return response
        except:
            context = {
                'error': True
            }
            return render(request, 'process.html', context)

    return render(request, 'process.html')
    