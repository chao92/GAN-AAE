import tensorflow as tf
import numpy as np
from config import Config



class EdgeGAN:
    def __init__(self, config, dropout=0.5, lr=0.00005):
        self.config = config
        self.dropout = dropout
        self.lr = lr
    def _weight_var(self, shape, name, dtype=tf.float32):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),dtype=dtype)

    def _bias_var(self, shape, name, dtype=tf.float32):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0),dtype=dtype)

    def y_to_probs(self,Y):
        return Y / tf.reduce_sum(Y, axis=-1, keep_dims=True)





    def create_discriminator_or_learner(self, name, x, y, device = "/gpu:0"):
        """
        TODO 调整计算方式适应Vertex的数量，压缩D_W1的数量
        :param x: batch_size * dimension
        :param y: batch_size * num_class
        :return:
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # Transform Y
            y = self.y_to_probs(y)
            num_class = self.config.num_class
            input_dim = self.config.x_dim
            dtype = self.config.dtype

            D_W_all = self._weight_var([num_class, input_dim * self.config.middle_size*2], 'Discriminator_all', dtype=dtype)
            D_b_all = self._bias_var([num_class, self.config.middle_size*2], 'Discriminator_b1', dtype=dtype)

            # batch * input_dim * self.config.middle_size
            D_W1 = tf.reshape(tf.matmul(y,D_W_all), [-1, input_dim, self.config.middle_size*2])
            D_b1 = tf.reshape(tf.matmul(y,D_b_all), [-1, self.config.middle_size*2])

            # Generate Parameter

            D_W2 = self._weight_var([self.config.middle_size*2, 1], 'Discriminator_W2', dtype=dtype)
            D_b2 = self._bias_var([1], 'Discriminator_b2', dtype=dtype)
            var_list = [D_W_all,D_b_all,D_W2,D_b2]
            x = tf.reshape(x, [-1, 1, input_dim])
            D_h1 = tf.nn.leaky_relu(tf.reshape(tf.matmul(x, D_W1), [-1, self.config.middle_size*2]) + D_b1)
            D_logit = tf.matmul(D_h1, D_W2) + D_b2
            D_prob = tf.nn.sigmoid(D_logit)

            return D_prob, var_list


    def create_generator(self, name, y, z, device = "/gpu:1"):
        with tf.variable_scope(name):
            z_dim = self.config.z_dim
            output_dim = self.config.x_dim
            num_class = self.config.num_class
            dtype = self.config.dtype

            y = self.y_to_probs(y)
            # z2 = W * z1 + b
            G_W2 = self._weight_var([self.config.middle_size, output_dim], 'G_W2', dtype=dtype)
            G_b2 = self._bias_var([output_dim], 'G_B2', dtype=dtype)


            # Transform Y
            G_W_all = self._weight_var([num_class, z_dim * self.config.middle_size], 'Generator_all', dtype=dtype)
            G_b_all = self._bias_var([num_class, self.config.middle_size], 'Generator_b1', dtype=dtype)

            var_list = [G_W_all, G_b_all, G_W2, G_b2]

            # batch * z_dim * self.config.middle_size
            G_W1 = tf.reshape(tf.matmul(y, G_W_all), [-1, z_dim, self.config.middle_size])
            G_b1 = tf.reshape(tf.matmul(y, G_b_all), [-1, self.config.middle_size])
            z = tf.reshape(z, [-1, 1, z_dim])
            G_h1 = tf.nn.leaky_relu(tf.nn.dropout(tf.reshape(tf.matmul(z, G_W1), [-1, self.config.middle_size]) + G_b1, keep_prob=self.dropout))
            G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
            G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob, var_list


    def _create_generator(self, name, y, z):
        """
        根据y生成X
        :param name:
        :param y:
        :return:
        """
        config = self.config
        with tf.variable_scope(name):
            dtype = config.dtype
            y_prob = self.y_to_probs(y)
            yz_concatenation = tf.concat([z, y_prob], axis=-1)
            W = self._weight_var([config.num_class+config.z_dim, config.x_dim], 'W', dtype=dtype)
            b = self._bias_var([config.x_dim], 'b', dtype=dtype)
            GX = tf.nn.leaky_relu(tf.matmul(yz_concatenation,W) + b)
            trainable_parameters = [W, b]

        return GX, trainable_parameters

    def create_classifer(self,name,x):
        """
        分类器
        :param name:
        :param x:
        :param y:
        :return:
        """
        config = self.config
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            dtype = config.dtype
            W = self._weight_var([config.x_dim, config.num_class], 'W', dtype=dtype)
            b = self._bias_var([config.num_class], 'b', dtype=dtype)
            # TODO +ReLU
            logits = tf.matmul(x,W) + b
            probs = tf.nn.softmax(logits, dim=-1)
            trainable_parameters = [W, b]
        return logits, probs, trainable_parameters

    def _sample_Z(self, m):
        '''Uniform prior for G(Z)'''
        return np.random.uniform(-1., 1., size=[m, self.config.z_dim])

    def clip_prob(self, x):
        return tf.clip_by_value(x, 1e-10, 1-1e-10)


    def build_graph(self):
        config = self.config


        """
        Graph
        """
        # TODO

        """
        Placeholders
        """
        self.global_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False)
        self.Xb = tf.placeholder(tf.bool, shape=[None, config.x_dim], name='X')
        self.Y = tf.placeholder(config.dtype, shape=[None, config.num_class], name='Y')
        self.Z = tf.placeholder(config.dtype, shape=[None, config.z_dim], name='Y')

        self.hb = tf.placeholder(tf.bool, shape=[None, config.x_dim], name='Head')
        self.ihb = tf.placeholder(tf.bool, shape=[None, config.x_dim], name='IncorrectHead')
        self.tb = tf.placeholder(tf.bool, shape=[None, config.x_dim], name='Tail')
        self.itb = tf.placeholder(tf.bool, shape=[None, config.x_dim], name='IncorrectTail')


        """
        Cast
        """
        self.X = tf.cast(self.Xb, config.dtype)
        self.h = tf.cast(self.hb, config.dtype)
        self.ih = tf.cast(self.ihb, config.dtype)
        self.t = tf.cast(self.tb, config.dtype)
        self.it = tf.cast(self.itb, config.dtype)

        """
        for Discrminator
        """

        # Maximize
        d_probs, d_paras = self.create_discriminator_or_learner("Discriminator", self.X, self.Y)
        classifier_logits, classifier_Y, c_paras = self.create_classifer("Classifier",self.X)
        l_probs, l_paras = self.create_discriminator_or_learner("Learner", self.X, classifier_Y)
        Generated_X, g_paras = self.create_generator("Generator",self.Y,self.Z)
        gd_probs, _ = self.create_discriminator_or_learner("Discriminator", Generated_X, self.Y)
        gl_probs, _ = self.create_discriminator_or_learner("Learner", Generated_X, self.Y)
        # Contrastive Loss


        def cosine(x,y):
            term1 = tf.reduce_sum(tf.multiply(x,y), axis=-1)
            term2 = tf.sqrt(tf.reduce_sum(tf.multiply(x,x), axis=-1))
            term3 = tf.sqrt(tf.reduce_sum(tf.multiply(y,y), axis=-1))
            res = term1/ (term3*term2)
            return res

        _, h_prob, _ = self.create_classifer("Classifier", self.h)
        _, ih_prob, _ = self.create_classifer("Classifier", self.ih)
        _, t_prob, _ = self.create_classifer("Classifier", self.t)
        _, it_prob, _ = self.create_classifer("Classifier", self.it)

        r = self.config.ratio
        m = self.config.margin
        self.contrasive_loss = r * tf.reduce_mean(1.0 - cosine(h_prob,t_prob)) + (1-r)*tf.reduce_mean(tf.maximum(0.0,cosine(ih_prob,it_prob)-m))
        prob_Y = self.y_to_probs(self.Y)

        discrminator_objective_term = - tf.log(self.clip_prob(d_probs)) - tf.log(self.clip_prob(1. - gd_probs))
        learner_objective_term = - tf.log(self.clip_prob(l_probs)) - tf.log(self.clip_prob(1. - gl_probs))
        generator_objective_term = - tf.log(self.clip_prob(gd_probs)) - tf.log(self.clip_prob(gl_probs))
        KL_term = tf.reduce_mean(tf.multiply(prob_Y, tf.log(tf.div(prob_Y, classifier_Y+ 1e-10) + 1e-10)))
        MSE_term = tf.reduce_mean(tf.square(prob_Y - classifier_Y))



        ECEE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=classifier_logits),axis=-1)
        CEE = tf.reduce_mean(-tf.reduce_sum(prob_Y* tf.log(classifier_Y ), axis=-1), axis=-1)
        # 定义损失和训练函数

        self.discriminator_loss = tf.reduce_mean(discrminator_objective_term )
        self.train_discriminator_op = self.optimize_with_clip(self.discriminator_loss, var_list=d_paras)
        self.learner_loss = tf.reduce_mean(learner_objective_term)
        self.train_learner_op = self.optimize_with_clip(self.learner_loss, var_list=l_paras)
        self.generator_loss = tf.reduce_mean(generator_objective_term)
        self.train_generator_op = self.optimize_with_clip(self.generator_loss, var_list=g_paras+d_paras+l_paras)
        self.classifier_loss = tf.reduce_mean(KL_term)
        self.train_classifier_op = self.optimize_with_clip(self.classifier_loss, var_list=c_paras, global_step=self.global_step)
        self.train_contrasive_op = self.optimize_with_clip(self.contrasive_loss, var_list=c_paras)
        # TODO Cosine 距离

        self.debug =  [] # [ECEE,CEE,prob_Y,self.Y]


        """
        For Inference
        """
        self.classifier_res = d_probs

    def optimize_with_clip(self, loss, var_list, global_step=None):
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads = optimizer.compute_gradients(loss=loss, var_list=var_list)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 1), v)  # clip gradients
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        return train_op

    def init_session(self, mode='Train'):
        print('initializing the model...')
        print('train_mode: %s' % mode)
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False  #: 是否打印设备分配日志
        config.allow_soft_placement = True  # ： 如果你指定的设备不存在，允许TF自动分配设备
        self.sess = tf.Session(config=config)

        # check from checkpoint
        ckpt_path = self.config.checkpoint_path
        print('check the checkpoint_path : %s' % ckpt_path)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        # TODO 把步数加入到其中
        if ckpt and ckpt.model_checkpoint_path:
            print('restoring from %s' % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        elif mode != 'Train':
            raise FileNotFoundError('Inference mode asks the checkpoint !')
        else:
            print('does not find the checkpoint, use fresh parameters')
            self.sess.run(tf.global_variables_initializer())

    def save_to_checkpoint(self, path=None):
        if path is None:
            path = self.config.checkpoint_path
        self.saver.save(self.sess, path + 'model.ckpt', global_step=self.global_step)
        print('checkpoint has been saved to :' + path + 'model.ckpt')

    """
    Training
    """
    def train_step(self,X_data, Y_data,h,t,ih,it):
        # Discriminator
        batch_size = self.config.batch_size
        Z = self._sample_Z(batch_size)
        _, discriminator_loss = self.sess.run([self.train_discriminator_op, self.discriminator_loss], feed_dict={
            self.Xb: X_data, self.Z: Z, self.Y: Y_data})
        _, learner_loss = self.sess.run([self.train_learner_op, self.learner_loss], feed_dict={
            self.Xb: X_data, self.Z: Z, self.Y: Y_data})
        _, generator_loss = self.sess.run([self.train_generator_op, self.generator_loss], feed_dict={
            self.Xb: X_data, self.Z: Z, self.Y: Y_data})
        _, classifier_loss = self.sess.run([self.train_classifier_op, self.classifier_loss], feed_dict={
            self.Xb: X_data, self.Z: Z, self.Y: Y_data})
        _, contrasive_loss = self.sess.run([self.train_contrasive_op, self.contrasive_loss], feed_dict={
            self.hb:h, self.ihb:ih, self.tb:t,self.itb:it})
        debug= self.sess.run([self.debug], feed_dict={
            self.Xb: X_data, self.Z: Z, self.Y: Y_data})
        step = self.sess.run(self.global_step)

        loss = [discriminator_loss,learner_loss,generator_loss,classifier_loss,contrasive_loss,debug]
        #loss = classifier_loss
        return step, loss

    """
    Infer or Test
    """

    def infer_step(self, X_data, Y_data):
        # Discriminator
        batch_size = self.config.batch_size
        Z = self._sample_Z(batch_size)
        probs = self.sess.run([self.classifier_res], feed_dict={
            self.Xb: X_data, self.Y: Y_data})
        return probs



