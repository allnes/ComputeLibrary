import argparse
import os
import numpy as np
import tensorflow as tf

def frozen_graph_maker(export_dir,output_graph):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                sess.graph_def,
                output_nodes# The output node names are used to select the usefull nodes
        )       
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
def main():
    parser = argparse.ArgumentParser('Extract TensorFlow net parameters')
    parser.add_argument('-m', dest='modelFile', type=str, required=True, help='Path to TensorFlow frozen graph file (.pb)')
    parser.add_argument('-d', dest='dumpPath', type=str, required=False, default='./', help='Path to store the resulting files.')
    parser.add_argument('--nostore', dest='storeRes', action='store_false', help='Specify if files should not be stored. Used for debugging.')
    parser.set_defaults(storeRes=True)
    args = parser.parse_args()
    export_dir=args.dumpPath
    output_graph = args.modelFile
    frozen_graph_maker(export_dir,output_graph)