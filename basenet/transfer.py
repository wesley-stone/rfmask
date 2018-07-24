from a3c_v2.a3c_v2 import *
from basenet.unet import *

def param_transfer(sess, fmodel_path, var_dict, fmeta='model.ckpt.meta'):
    gtmp = tf.Graph()
    fvar_name = []
    with gtmp.as_default(), tf.Session(graph=gtmp) as stmp:
        tf.train.import_meta_graph(os.path.join(fmodel_path, fmeta))
        for v in tf.trainable_variables():
            fvar_name.append(v.op.name)

    # for fv in fvar_name:
        # print(fv)
    # print('======================================================')
    vdict = dict()
    tensor_set = set()
    for (vname, tensor) in var_dict.items():
        for fvn in fvar_name:
            if vname.find(fvn) >= 0:
                vdict[fvn] = tensor
                # print(vdict[fvn])
                tensor_set.add(tensor)
    # print('======================================================')
    s = tf.train.Saver(var_list=vdict)
    s.restore(sess, tf.train.latest_checkpoint(fmodel_path))
    return tensor_set


def guarantee_initialized_variables(session, inited, list_of_variables = None):
    if list_of_variables is None:
        list_of_variables = tf.global_variables()
    if inited is None:
        session.run(tf.variables_initializer(list_of_variables))
    else:
        uninited = []
        for v in list_of_variables:
            if v not in inited:
                uninited.append(v)
                print(v)
        if len(uninited):
            session.run(tf.variables_initializer(uninited))


if __name__ == '__main__':
    AC = AC_Network((512, 512), 'global', None)
    sess = tf.Session()
    param_transfer(sess, './unet_trained/', AC.unet_var_dict)
    sess.run(AC.unet_var_dict)
    print('==========================================')








