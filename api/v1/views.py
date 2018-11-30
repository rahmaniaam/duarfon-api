from rest_framework.generics import RetrieveAPIView
from rest_framework.response import Response
from ml.Final import run

class PredictView(RetrieveAPIView):

  def get(self, request):
    result = run()
    return Response(data=result)
