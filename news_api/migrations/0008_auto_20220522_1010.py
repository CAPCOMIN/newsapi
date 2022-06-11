# Generated by Django 3.1.7 on 2022-05-22 10:10

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('news_api', '0007_auto_20210330_2342'),
    ]

    operations = [
        migrations.CreateModel(
            name='givelike',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('userid', models.IntegerField()),
                ('newsid', models.IntegerField()),
                ('givelikeornot', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='message',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('userid', models.IntegerField()),
                ('message', models.CharField(max_length=1000)),
                ('time', models.CharField(max_length=30)),
                ('newsid', models.IntegerField()),
                ('hadread', models.IntegerField()),
                ('title', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='spiderstate',
            fields=[
                ('spiderid', models.IntegerField(primary_key=True, serialize=False)),
                ('status', models.IntegerField()),
                ('interval', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='urlcollect',
            fields=[
                ('url', models.CharField(max_length=255, primary_key=True, serialize=False)),
                ('handle', models.IntegerField()),
                ('type', models.IntegerField()),
                ('time', models.CharField(max_length=30)),
            ],
        ),
        migrations.RemoveField(
            model_name='newshot',
            name='id',
        ),
        migrations.RemoveField(
            model_name='newssimilar',
            name='id',
        ),
        migrations.RemoveField(
            model_name='recommend',
            name='id',
        ),
        migrations.AddField(
            model_name='comments',
            name='status',
            field=models.CharField(default=1, max_length=20),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='newsdetail',
            name='keywords',
            field=models.CharField(default=1, max_length=1000),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='recommend',
            name='time',
            field=models.CharField(default=django.utils.timezone.now, max_length=30),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='user',
            name='region',
            field=models.CharField(default=1, max_length=30),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='newshot',
            name='news_id',
            field=models.IntegerField(primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='newssimilar',
            name='new_id_base',
            field=models.CharField(max_length=64, primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='recommend',
            name='userid',
            field=models.IntegerField(primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='user',
            name='tags',
            field=models.CharField(max_length=2000),
        ),
        migrations.AlterField(
            model_name='user',
            name='tagsweight',
            field=models.CharField(max_length=2000),
        ),
    ]
