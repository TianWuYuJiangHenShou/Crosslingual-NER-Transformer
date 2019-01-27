import tensorflow as tf
import tensorflow_hub as hub
import math

import numpy as np
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

model_dir = 'data/output_model_config_fb_nalign_layer_e_e0_enzh_new_new'
#fout = open('results/demo_hidden/trans.txt', 'w')


def read_dict(filename):
	d = {}
	with open(filename) as f:
		for i, line in enumerate(f):
			d[line.strip()] = i
	return d

def get_sent(filename, d):
	sents = []
	sent = []
	sents_id = []
	sent_id = []
	with open(filename) as f:
		for line in f.readlines():
			line = line.strip()
			if len(line) == 0:
				if len(sent) == 0:
					continue
				sents.append(sent)
				sents_id.append(sent_id)
				sent, sent_id = [], []
			else:
				token = line.split()[0]
				sent.append(token)
				token = token.lower()
				if token in d:
					sent_id.append(d[token])
				else:
					sent_id.append(d['<unk>'])
		if len(sent) != 0:
			sents.append(sent)
			sents_id.append(sent_id)
	return sents, sents_id


def shape_list(x):
	ps = x.get_shape().as_list()
	ts = tf.shape(x)
	return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def get_sent_wiki(filename, d):
	sents_id = []
	sents = []
	with open(filename) as f:
		for line in f.readlines():
			line = line.strip()
			if len(line) == 0:
				continue
			sent = line.split(' ')
			sents.append(sent)
			sent_id = []
			for token in sent:
				token = token.lower()
				if token in d:
					sent_id.append(d[token])
				else:
					sent_id.append(d['<unk>'])
			sents_id.append(sent_id)
	return sents, sents_id

def generate_1sent(filename, d, wid, output, sess):
	with open(filename) as f:
		for line in f.readlines():
			line = line.strip()
			if len(line) == 0:
				continue
			sent = line.split(' ')
			sent_id = []
			for token in sent:
				token = token.lower()
				if token in d:
					sent_id.append(d[token])
				else:
					sent_id.append(d['<unk>'])
			vec = sess.run(output, feed_dict={wid: [sent_id]})
			yield sent, vec[0]

def cal_high1_big():
        fout = open('results/demo_hidden/layer5_high1_full.txt', 'w')
        word_count = 0
        bases = ['black', 'brown', 'chair']
        for i, vsrc_sent in enumerate(src_rep):
                for ii, vsrc in enumerate(vsrc_sent):
                        word = sent_src[i][ii]
                        if 'black' not in word and 'brown' not in word and 'chair' not in word:
                                continue
                        max_score = float('-inf')
                        max_j, max_jj = -1, -1
                        word_cal = 0
			#for sent_tgt, vtgt_sent in generate_1sent('data/demo_hidden/eswiki_train.seg', dict_tgt, wid_tgt, output_tgt, sess):
                        for j, sent_id in enumerate(sent_id_tgt):
                                vec = sess.run(output_tgt, feed_dict={wid_tgt: [sent_id]})
                                for jj, vtgt in enumerate(vec[0]):
                                        cos_sim = sum(vsrc * vtgt) / math.sqrt(sum(vsrc**2) * sum(vtgt**2))
                                        if cos_sim > max_score:
                                                max_score, max_j, max_jj = cos_sim, j, jj
                                        word_cal += 1
                                        if word_cal % 1000000 == 0:
                                               print('{} words calculated'.format(word_cal))
                        word_count += 1
                        print('{} words have been processed'.format(word_count))
                        fout.write('{} {} {}\n'.format(sent_src[i][ii], sent_tgt[max_j][max_jj], max_score))
                        #fout.flush()
                        for k, token in enumerate(sent_src[i]):
                                if k == ii:
                                        fout.write('|{}| '.format(token))
                                else:
                                        fout.write(token + ' ')
                        fout.write('\n')
                        for k, token in enumerate(sent_tgt[max_j]):
                                if k == max_jj:
                                        fout.write('|{}| '.format(token))
                                else:
                                        fout.write(token + ' ')
                        fout.write('\n\n')
                        fout.flush()
                        #fout.write(' '.join(sent_src[i]) + '\n')
                        #fout.write(' '.join(sent_tgt[max_j]) + '\n\n')
        fout.close()

def plot_tsne(src_vec, tgt_vec):
	all_vec = np.asarray(src_vec + tgt_vec)
	embeded = TSNE(n_components=2).fit_transform(all_vec)
	print(embeded.shape)
	#with open('results/demo_hidden/tsne/all_vec.txt', 'w') as f:
	fsrc = open('results/demo_hidden/tsne/ontonote_sample/layer6_src_vec.txt', 'w')
	#fsrc_emp = open('results/demo_hidden/tsne/noalign/layer3_src_emp.txt', 'w')
	ftgt = open('results/demo_hidden/tsne/ontonote_sample/layer6_tgt_vec.txt', 'w')
	for i in range(len(all_vec)):
		if i < len(src_vec):
			#if i in src_emp:
			#	fsrc_emp.write('{} {}\n'.format(embeded[i][0], embeded[i][1]))
			#else:
			fsrc.write('{} {}\n'.format(embeded[i][0], embeded[i][1]))
		else:
			ftgt.write('{} {}\n'.format(embeded[i][0], embeded[i][1]))



if __name__ == '__main__':
	dict_src = read_dict('data/transformer_en_es_1B_vocab.20w')
	#dict_tgt = read_dict('data/transformer_en_es_eswiki_vocab.20w')
	dict_tgt = read_dict('data/transformer_en_zh_zhwiki_vocab.20w')
	#sent_src, sent_id_src = get_sent_wiki('data/demo_hidden/src4.txt', dict_src)
	#sent_src, sent_id_src = get_sent_wiki('data/demo_hidden/enwiki.1000', dict_src)
	sent_src, sent_id_src = get_sent('data/demo_hidden/ontonote4_en.sample', dict_src)
	"""
	src_samples = [234, 1066, 4, 2495, 2920]
	sent_src = [sent_src[n] for n in src_samples]
	sent_id_src = [sent_id_src[n] for n in src_samples]
	"""
	#print([len(sent) for sent in sent_id_src])
	#for i in range(5):
	#	print(sent_src[i])
	#sent_src = sent_src[10, ]
	#sent_id_src = sent_id_src[:200]
	#sent_src = sent_src[:200]
	#sent_src = np.asarray(sent_src)
	#print(' '.join(sent_src[0]))
	#exit(0)
	#print(sent_src)
	#print(sent_id_src)
	#sent_tgt, sent_id_tgt = get_sent_wiki('data/demo_hidden/tgt4.txt', dict_tgt)
	#sent_tgt, sent_id_tgt = get_sent_wiki('data/demo_hidden/eswiki.1000', dict_tgt)
	sent_tgt, sent_id_tgt = get_sent('data/demo_hidden/ontonote4_zh.sample', dict_tgt)
	"""
	tgt_samples = [232, 1067, 774, 513, 1352]
	sent_tgt = [sent_tgt[n] for n in tgt_samples]
	sent_id_tgt = [sent_id_tgt[n] for n in tgt_samples]
	"""
	#print([len(sent) for sent in sent_id_tgt])
	#exit(0)
	#sent_tgt = sent_tgt[:200]
	#sent_id_tgt = sent_id_tgt[:200]
	#sent_tgt = [sent_tgt[2]]
	#sent_id_tgt = [sent_id_tgt[2]]
	#sent_tgt = np.asarray(sent_tgt)
	#print(sent_tgt)
	#print(sent_id_tgt)
	#exit(0)
	wid_src = tf.placeholder(tf.int64, shape=[None, None])
	wid_tgt = tf.placeholder(tf.int64, shape=[None, None])
	print('Using transformer model from:', model_dir)
	layer = 6
	transformer_model = hub.Module(model_dir)
	output_src = transformer_model(wid_src, signature='en')
	output_tgt = transformer_model(wid_tgt, signature='zh')
	
	seq_len = shape_list(output_src)[1]
	output_src = tf.reshape(output_src, shape=[-1, seq_len, 2, 7, 512])
	output_src = tf.reshape(output_src[:, :, :, layer, :], shape=[-1, seq_len, 2*512])
	seq_len = shape_list(output_tgt)[1]
	output_tgt = tf.reshape(output_tgt, shape=[-1, seq_len, 2, 7, 512])
	output_tgt = tf.reshape(output_tgt[:, :, :, layer, :], shape=[-1, seq_len, 2*512])
	

	src_rep, tgt_rep = [], []

	all_vec = []
	src_vec = []
	tgt_vec = []
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		
		for sent_id in sent_id_src:
			src_rep_sent = sess.run(output_src, feed_dict={wid_src: [sent_id]})
			#print(len(src_rep_sent[0]))
			src_rep.append(src_rep_sent[0])
			all_vec.extend(src_rep_sent[0])
			src_vec.extend(src_rep_sent[0])
                

		#cal_high1_big()
		#exit(0)

		for sent_id in sent_id_tgt:
			tgt_rep_sent = sess.run(output_tgt, feed_dict={wid_tgt: [sent_id]})
			#print(len(tgt_rep_sent[0]))
			tgt_rep.append(tgt_rep_sent[0])
			all_vec.extend(tgt_rep_sent[0])
			tgt_vec.extend(tgt_rep_sent[0])
		#src_rep, tgt_rep = sess.run([output_src, output_tgt], feed_dict={wid_src: sent_id_src, wid_tgt: sent_id_tgt})
	
	print('Total {} source sent processed.'.format(len(src_rep)))
	print('Total {} target sent processed.'.format(len(tgt_rep)))

	"""
	src_emp = []
	count = 0
	for sent in sent_src:
		for w in sent:
			if w == 'brown':
				src_emp.append(count)
			count += 1
	print(src_emp)
	"""
	
	print(len(src_vec))
	print(len(tgt_vec))
	plot_tsne(src_vec, tgt_vec)
	exit(0)
	"""
	N=5
	all_vec = np.asarray(all_vec)
	estimator = KMeans(n_clusters=N)
	estimator.fit(all_vec)

	result_id = [[] for i in range(N)]
	#result_id_tgt = [[] for i in range(N)]
	count = 0
	for i, sent_id in enumerate(sent_id_src + sent_id_tgt):
		for j in range(len(sent_id)):
			result_id[estimator.labels_[count]].append((i, j))
			count += 1

	#fout = open('results/demo_hidden/layer4_kmeans_5.txt', 'w', encoding='utf8')
	for nc, cluster in enumerate(result_id):
		fout = open('results/demo_hidden/kmeans_big/layer3_kmeans5_cluster{}.txt'.format(nc), 'w', encoding='utf8')
		fout.write('### Cluster {}\n'.format(nc))
		for item in cluster:
			if item[0] >= len(sent_src):
				fout.write('{}: {}\n'.format(sent_tgt[item[0] - len(sent_src)][item[1]], ' '.join(sent_tgt[item[0] - len(sent_src)])))
			else:
				fout.write('{}: {}\n'.format(sent_src[ item[0] ][ item[1] ], ' '.join(sent_src[ item[0] ])))
		fout.write('\n')
		fout.close()
	"""

	
	fout = open('results/demo_hidden/ontonote_zh_sample.txt', 'w')
	#for i, vsrc_sent in enumerate(src_rep):
	for i, vsrc_sent in enumerate(tgt_rep):
		for j, vtgt_sent in enumerate(tgt_rep):
			fout.write('\t' + '\t'.join(sent_tgt[j]) + '\n')
			matrix = []
			for ii, vsrc in enumerate(vsrc_sent):
				#fout.write(sent_src[i][ii])
				a = []
				for jj, vtgt in enumerate(vtgt_sent):	
					cos_sim = sum(vsrc * vtgt) / math.sqrt(sum(vsrc**2) * sum(vtgt**2))
					#euc_dis = math.sqrt(sum((vsrc - vtgt) ** 2))
					#fout.write('{}\t{}\t{}\n'.format(sent_src[0][i], sent_tgt[0][j], cos_sim))
					#a.append(cos_sim)
					a.append(cos_sim)
				matrix.append(a)
				
				#m = max(a)
				#for score in a:
				#	if score == m:
				#		fout.write('\t{:.3f}'.format(score))
				#	else:
				#		fout.write('\t{:.3f}'.format(score))
				#fout.write('\n')
				
			matrix = np.asarray(matrix)
			
			csls = False
			if csls:
				K=5
				src_knn = [np.mean(sorted(m, reverse=True)[:K]) for m in matrix]
				tgt_knn = [np.mean(sorted(matrix[:, k], reverse=True)[:K]) for k in range(len(vtgt_sent))]

				for ii in range(len(vsrc_sent)):
					for jj in range(len(vtgt_sent)):
						matrix[ii][jj] = 2 * matrix[ii][jj] - src_knn[ii] - tgt_knn[jj]

			#matrix = np.exp(matrix) / np.sum(np.exp(matrix), 0)
			#for ii, token in enumerate(sent_src[i]):
			for ii, token in enumerate(sent_tgt[i]):
				fout.write(token)
				for score in matrix[ii]:
					fout.write('\t{:.3f}'.format(score))
				#fout.write('\t'.join(matrix[ii]))
				fout.write('\n')
			fout.write('\n')
	fout.close()
	

	"""
	fout = open('results/demo_hidden/recall_high1/layer3_high1_sample.txt', 'w')
	word_count = 0
	bases = ['black', 'brown', 'chair']
	for i, vsrc_sent in enumerate(src_rep):
		for ii, vsrc in enumerate(vsrc_sent):
			word = sent_src[i][ii]
			if 'black' not in word and 'brown' not in word and 'chair' not in word:
				continue
			max_score = float('-inf')
			max_j, max_jj = -1, -1
			word_cal = 0
			for j, vtgt_sent in enumerate(tgt_rep):
				for jj, vtgt in enumerate(vtgt_sent):
					cos_sim = sum(vsrc * vtgt) / math.sqrt(sum(vsrc**2) * sum(vtgt**2))
					if cos_sim > max_score:
						max_score, max_j, max_jj = cos_sim, j, jj
					word_cal += 1
					#if word_cal % 100 == 0:
					#	print('{} words calculated'.format(word_cal))
			word_count += 1
			print('{} words have been processed'.format(word_count))
			fout.write('{} {} {}\n'.format(sent_src[i][ii], sent_tgt[max_j][max_jj], max_score))
			#fout.flush()
			for k, token in enumerate(sent_src[i]):
				if k == ii:
					fout.write('|{}| '.format(token))
				else:
					fout.write(token + ' ')
			fout.write('\n')
			for k, token in enumerate(sent_tgt[max_j]):
				if k == max_jj:
					fout.write('|{}| '.format(token))
				else:
					fout.write(token + ' ')
			fout.write('\n\n')
			fout.flush()
			#fout.write(' '.join(sent_src[i]) + '\n')
			#fout.write(' '.join(sent_tgt[max_j]) + '\n\n')
	
	fout.close()
	"""
		
