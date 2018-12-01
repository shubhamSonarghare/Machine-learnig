import tensorflow as tf
import cv2
from PIL import Image


def load_opt_graph(opt_graph):
  with tf.gfile.GFile(opt_graph, 'rb') as f:
   graph_def_optimized = tf.GraphDef()
   graph_def_optimized.ParseFromString(f.read())
  with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def_optimized)
  return graph
  
dictr = {'1':'fwd','2':'left','3':'right','4':'stop'}
gr = load_opt_graph('model\\optimized_graph.pb')


for op in gr.get_operations():
  print(op.name)

ip = gr.get_tensor_by_name('import/MyInput:0')
prediction = gr.get_tensor_by_name('import/myOutput:0')

img = cv2.imread('Test images/image1.jpg')
resized = cv2.resize(img, (40,20), interpolation = cv2.INTER_AREA)
im = Image.fromarray(resized)
net_img = np.reshape(resized,[resized.shape[0]*resized.shape[1]*resized.shape[2],])
prec = 0
with tf.Session(graph=gr) as sess:
  op = sess.run(prediction, feed_dict={ip:net_img})
  prec = np.argmax(op)


plt.figure()

plt.imshow(img, interpolation="nearest", cmap="gray", )
plt.title('Predicted val: '+str(prec) + ' i.e move ' + str(dictr[str(prec)]))
plt.show()



'''
########################## for Video Processing #####################
dictr = {'1':'fwd','2':'left','3':'right','4':'stop'}
cap = cv2.VideoCapture('more_test/VID_20181023_141446.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out_vid = cv2.VideoWriter('op.avi',cv2.VideoWriter_fourcc('M','J','P','G'),16.633,(frame_width,frame_height))
cnt=0
with tf.Session(graph=gr) as sess:
  
  while(cap.isOpened()):
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    z= frame[:]
    resized = cv2.resize(z, (40,20), interpolation = cv2.INTER_AREA)
    net_img = np.reshape(resized,[resized.shape[0]*resized.shape[1]*resized.shape[2],])
    op = sess.run(prediction, feed_dict={ip:net_img})
    prec = np.argmax(op)
    #print('Predicted val: ',prec)
    cv2.putText(frame, 'Predicted val: '+dictr[str(prec)], (700, 50), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('vid',frame)
    out_vid.write(frame)
    #cv2.title('Predicted val: '+str(prec)+str(net_img.shape))
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

cap.release()
out_vid.release()
cv2.destroyAllWindows()

'''
