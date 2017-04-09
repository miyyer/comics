import theano, lasagne
import numpy as np
import h5py as h5
import theano.tensor as T   

'''HELPER METHODS'''
def softmax_2d(x):
    e = T.exp(x - T.max(x, axis=-1)[:, None])
    dist = e / T.sum(e, axis=-1)[:, None]
    return dist

# combine text and image data into tuple
def load_hdf5(fold_data, fold_vgg):
    img_mask = fold_data['panel_mask']
    book_ids = fold_data['book_ids']
    page_ids = fold_data['page_ids']
    bboxes = fold_data['bbox']
    bbox_mask = fold_data['bbox_mask']
    words = fold_data['words']
    word_mask = fold_data['word_mask']
    data = (img_mask, book_ids, page_ids, bboxes, bbox_mask, words, \
        word_mask, fold_vgg['vgg_features'])
    return data


'''LAYERS'''
class MyConcatLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, is_answer=False, axis=1, **kwargs):
        super(MyConcatLayer, self).__init__(incomings, **kwargs)
        self.axis = axis
        self.is_answer = is_answer

    def get_output_shape_for(self, input_shapes):

        if self.is_answer:
            return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2] + input_shapes[1][1])

        else:
            output_shape = [next((s for s in sizes if s is not None), None)
                            for sizes in zip(*input_shapes)]

            def match(shape1, shape2):
                axis = self.axis if self.axis >= 0 else len(shape1) + self.axis
                return (len(shape1) == len(shape2) and
                        all(i == axis or s1 is None or s2 is None or s1 == s2
                            for i, (s1, s2) in enumerate(zip(shape1, shape2))))

            # Check for compatibility with inferred output shape
            if not all(match(shape, output_shape) for shape in input_shapes):
                raise ValueError("Mismatch: input shapes must be the same except "
                                 "in the concatenation axis")
            # Infer output shape on concatenation axis and return
            sizes = [input_shape[self.axis] for input_shape in input_shapes]
            concat_size = None if any(s is None for s in sizes) else sum(sizes)
            output_shape[self.axis] = concat_size
            return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        if self.is_answer:
            reshaped_answers = inputs[1].reshape((inputs[1].shape[0], 1, inputs[1].shape[1]), ndim=3)     
            repeated_answers = T.repeat(reshaped_answers, 3, axis=1)                                      
            return T.concatenate([inputs[0], repeated_answers], axis=-1)   
        else:
            return T.concatenate(inputs, axis=self.axis)


# compute vector average
class SumAverageLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, compute_sum=True, num_dims=3, **kwargs):
        super(SumAverageLayer, self).__init__(incomings, **kwargs)
        self.sum = compute_sum
        self.num_dims = num_dims

    def get_output_for(self, inputs, **kwargs):
        if self.num_dims == 3:
            emb_sums = T.sum(inputs[0] * inputs[1][:, :, :, None], axis=2)
            if self.sum:
                return emb_sums
            else:
                mask_sums = T.sum(inputs[1], axis=2)
                return emb_sums / mask_sums[:, :, None]
        
        elif self.num_dims == 4:
            emb_sums = T.sum(inputs[0] * inputs[1][:, :, :, :, None], axis=3)
            if self.sum:
                return emb_sums
            else:
                mask_sums = T.sum(inputs[1], axis=3)
                return emb_sums / mask_sums[:, :, :, None]

    def get_output_shape_for(self, input_shapes):
        if self.num_dims == 3:
            return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][-1])
        elif self.num_dims == 4:
             return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2], input_shapes[0][-1])


# score candidate answers by their inner product with context
class InnerProductLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, is_cc=False, f=softmax_2d, **kwargs):
        super(InnerProductLayer, self).__init__(incomings, **kwargs)
        self.f = f 
        self.is_cc = is_cc

    def get_output_for(self, inputs, **kwargs):
        if self.is_cc:
            prod = T.sum(inputs[0] * inputs[1], axis=-1, keepdims=True)
        else:
            prod = T.sum(inputs[0][:, None, :] * inputs[1], axis=-1)
            prod = self.f(prod)
        return prod

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1][:-1]
