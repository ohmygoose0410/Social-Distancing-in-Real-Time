# Social-Distancing-in-Real-Time

## 運行方式

1.  先透過下面指令使用滑鼠將要偵測之區域圈出來，程式會做透視變換把4邊形轉換成正方形或是矩形，再來判斷曼哈頓距離。

    ```bash
    $ pip install pyyaml
    $ python correction.py --video video/AVG-TownCentre-raw.webm -H 300 -S 500
    ```

    > [!WARNING]
    > 在使用滑鼠圈出偵測區域位置時，必須依照右上、右下、左下和左上之順序做標註。

2.  接續執行main.py即可。

    ```bash
    $ python main.py --video video/AVG-TownCentre-raw.webm
    ```

## Demo影片
    
<img src="output/social distance.gif" width="600">
