from django.apps import AppConfig
from django.db.models.signals import post_migrate

def increment_callback(sender, **kwargs):
    from news_api.models import newsdetail

    if sender.name == 'news_api':
        try:
            test_model = newsdetail.objects.create(id=1)
            test_model.delete()
        except:
            pass

class NewsApiConfig(AppConfig):
    name = 'news_api'

    def ready(self):
        # 注册改回调函数
        post_migrate.connect(increment_callback, sender=self)
