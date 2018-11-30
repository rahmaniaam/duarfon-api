from rest_framework.generics import RetrieveAPIView
from rest_framework.response import Response

class PredictView(RetrieveAPIView):

  def get(self, request):
    return Response(data={"wow": 3})
