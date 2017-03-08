
### 阅遍了近4万首唐诗

```
龙舆迎池里，控列守龙猱。
几岁芳篁落，来和晚月中。
殊乘暮心处，麦光属激羁。
铁门通眼峡，高桂露沙连。
倘子门中望，何妨嶮锦楼。
择闻洛臣识，椒苑根觞吼。
柳翰天河酒，光方入胶明
```

这诗做的很有感觉啊，这都是勤奋的结果啊，基本上学习了全唐诗的所有精华才有了这么牛逼的能力，这一般人能做到？本博客讲讲解一些里面实现的技术细节，如果有未尽之处，大家可以通过微信找到我，那个头像很神奇的男人。
闲话不多说，先把github链接放上来，这个作诗机器人我会一直维护，如果大家因为时间太紧没有时间看，可以给这个项目star一下或者fork，我一推送更新你就能看到，主要是为了修复一些api问题，tensorflow虽然到了1.0，但是api还是会变化。
把星星加起来，让更多人可以看到我们创造这个作诗机器人，后期会加入更多牛逼掉渣天的功能，比如说押韵等等。
![机器人做的古诗](http://ofwzcunzi.bkt.clouddn.com/LSRjdHmHhOWemUmd.png)
![机器人做的古诗](http://ofwzcunzi.bkt.clouddn.com/ZHMKE1xE9S3Q3tOo.png)
![机器人做的古诗](http://ofwzcunzi.bkt.clouddn.com/EZPgxTi5ZBrBXK8p.png)

### Install tensorflow_poems

* 安装要求：
```
tensorflow 1.0
python3.5
all platform
```

* 安装作诗机器人， 简单粗暴，一顿clone：
```
git clone https://github.com/jinfagang/tensorflow_poems.git
```
由于数据大小的原因我没有把数据放到repo里面，大家🏠我的QQ： 1195889656 或者微信： jintianiloveu 我发给你们把，顺便给我们的项目点个赞哦！～

* 使用方法：
```
# for train
python3 tang_poems.py train

# for generate
python3 tang_poems.py gen
```

训练的时候有点慢，有GPU就更好啦，最后gen的时候你就可以看到我们牛逼掉渣天的诗啦！

```
龙舆迎池里，控列守龙猱。
几岁芳篁落，来和晚月中。
殊乘暮心处，麦光属激羁。
铁门通眼峡，高桂露沙连。
倘子门中望，何妨嶮锦楼。
择闻洛臣识，椒苑根觞吼。
柳翰天河酒，光方入胶明
```
感觉有一种李白的豪放风度！

### Author & Cite
This repo implement by Jin Fagang.
(c) Jin Fagang.
Blog: jinfagang.github.io
