# Generated by Django 3.1.7 on 2022-06-11 23:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('news_api', '0008_auto_20220522_1010'),
    ]

    operations = [
        migrations.AlterField(
            model_name='newsdetail',
            name='news_id',
            field=models.IntegerField(primary_key=True, serialize=False),
        ),
    ]
