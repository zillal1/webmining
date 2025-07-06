numpy              1.26.3
torch              2.3.0+cu118
torchaudio         2.3.0+cu118
torchtext          0.18.0
torchvision        0.18.0+cu118
torchtext          0.18.0

如何使用scrap_amazon.ipynb:
将文件中的your phone number here和your password here换成自己的，用来登录amazon
命令行运行Start-Process powershell -ArgumentList "-NoProfile -Command `"jupyter nbconvert --to notebook --execute scrap_amazon.ipynb --output executed_scrap_amazon.ipynb *> log.txt`""，并且电脑设置成永远不进入睡眠状态、永远不休眠，就能够在后台运行爬虫程序啦
可能遇到库没装的问题，自己按照提示装就行