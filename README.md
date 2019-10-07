This is a tiny python tool to visualize loss when training machine learning model.

Principally, it can be used for any value monitor in Jupyter Interaction Interface.

The progress bar script base on the FastAi repository: [fastprogress](https://github.com/fastai/fastprogress), but reimplemented for multiply tracking graph.

![mul_fast](/home/tianning/Documents/MachineLearning/TRACKING/standard_file/images/mul_fast.gif)

```python
# using set_multiply_graph(nrow=1, ncol=2, figsize=(6,6)) 
# to initial the image layout and size control 
mb.set_multiply_graph(figsize=(12,4));
# names = [[name for curve,name for curve],[name for curve]]
mb.names = [['cos','sin'], ['sin']]
for i in mb:
    x = np.arange(0, 2*(i+1)*np.pi/10, 0.01)
    y1, y2 = np.cos(x), np.sin(x)
    # graphs = [subgraph_1,subgraph_2,...]
    # subgraph=[curve_a,curve_b,...]
    graphs = [[[x,y1],[x,y2]], [[x,y2]]]
    # bounds = [bounds for subgraph_1,bounds for subgraph_2,...]
    x_bounds = [None,None]
    y_bounds = [[-1,1],None]
    mb.update_graph_multiply(graphs,x_bounds,y_bounds)
mb.update_graph_multiply(graphs,x_bounds,y_bounds)
```

MLlog.py file provide a good log tracking framwork.

There are two units for log system:

- Meter Machine: the processer for loss curve, return a statistic loss

  - IdentyMeter   : Do nothing
  - AverageMeter: take average of last 100 loss outputs, so the loss curve will be more smooth.
  - ......

- Curve_data(max_leng = np.inf,Reduce=False)

  It define the style of drawed picture, for example:

  - Curve_data() will store all the statistic loss and draw.

  - Curve_data(100) mean the loss curve will update per 100 loss.
  - Curve_data(Reduce=True) mean it will reduce sample `self. reducer=100` data points when all data points hit  `self.reduce_limit = 10000`

```python
from MLlog import AverageMeter,RecordLoss,Curve_data,IdentyMeter
log_hand= RecordLoss(loss_records=[IdentyMeter(),IdentyMeter()],\
                     graph_set   =[Curve_data(100),Curve_data(Reduce=True)])
#----------code---------------
#----------code---------------
#----------code---------------
#loss = .......
log_hand.step(steps,[loss,loss])
log_hand.update_graph(mb,steps)
```

![MLlog](/home/tianning/Documents/MachineLearning/TRACKING/standard_file/images/MLlog.gif)