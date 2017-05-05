# -*- coding: utf-8 -*-
"""
| **@created on:** 20/05/16,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| File Utilities
|
| Sphinx Documentation Status:** Complete
|
..todo::
"""

from ast import literal_eval
from typeguard import typechecked
import numpy as np
import pandas as pd
from pyhdfs import HdfsClient
import operator
import functools
from typing import Union
import logging

logger = logging.getLogger(__name__)


def convert_df_tolist(*input_data):
    """
    | **@author:** Prathyush SP
    |
    | Convert Dataframe to List
    |
    :param input_data: Input Data (*args)
    :return: Dataframe
    .. todo::
        Prathyush SP:
            1. Check list logic
            2. Perform Dataframe Validation
    """
    dataframes = []
    for df in input_data:
        if len(df) == 0:
            continue
        else:
            if isinstance(df, pd.DataFrame):
                if len(input_data) == 1:
                    return df.values.tolist()
                dataframes.append(df.values.tolist())
            elif isinstance(df, pd.Series):
                df_list = df.to_frame().values.tolist()
                if isinstance(df_list, list):
                    if isinstance(df_list[0][0], list):
                        dataframes.append([i[0] for i in df.to_frame().values.tolist()])
                    else:
                        dataframes.append(df.to_frame().values.tolist())
                else:
                    dataframes.append(df.to_frame().values.tolist())
    return dataframes


@typechecked()
def read_csv(filename: Union[str, object] = None, data: Union[list, np.ndarray, pd.DataFrame] = None,
             split_ratio: Union[list, None] = (50, 20, 30), delimiter: str = ',', normalize: bool = False, dtype=None,
             header: Union[bool, int, list] = None, skiprows: int = None, ignore_cols=None, select_cols: list = None,
             index_col: int = False, output_label: Union[str, int, list, bool] = True, randomize: bool = False,
             return_as_dataframe: bool = False, describe: bool = False, label_vector: bool = False):
    """
    | **@author:** Prathyush SP
    |
    | The function is used to read a csv file with a specified delimiter
    :param filename: File name with absolute path
    :param data: Data used for train and test
    :param split_ratio: Ratio used to split data into train and test
    :param delimiter: Delimiter used to split columns
    :param normalize: Normalize the Data
    :param dtype: Data Format
    :param header: Column Header
    :param skiprows: Skip specified number of rows
    :param index_col: Index Column
    :param output_label: Column which specifies whether output label should be available or not.
    :param randomize: Randomize data
    :param return_as_dataframe: Returns as a dataframes
    :param describe: Describe Input Data
    :param label_vector: True if output label is a vector
    :return: return train_data, train_label, test_data, test_label based on return_as_dataframe
    """
    header = 0 if header else None
    if filename:
        df = pd.read_csv(filename, sep=delimiter, index_col=index_col, header=header, dtype=dtype, skiprows=skiprows)
    elif isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, np.ndarray) or isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise Exception('Filename / Data are None. Specify atleast one source')
    if describe:
        print(df.describe())
    df = df.sample(frac=1) if randomize else df
    df = df.apply(lambda x: np.log(x)) if normalize else df
    if len(split_ratio) == 3 and sum(split_ratio) == 100:
        test_size = int(len(df) * split_ratio[-1] / 100)
        valid_size = int(len(df) * split_ratio[1] / 100)
        train_size = int(len(df) - (test_size + valid_size))
        train_data_df, valid_data_df, test_data_df = df.head(int(train_size)), \
                                                     df.iloc[train_size:(train_size + valid_size)], \
                                                     df.tail(int(test_size))
        if output_label is None or output_label is False:
            if ignore_cols:
                ignore_cols = check_ignore_cols(ignore_cols)
                train_data_df = train_data_df.drop(ignore_cols, axis=1)
                test_data_df = test_data_df.drop(ignore_cols, axis=1)
                valid_data_df = valid_data_df.drop(ignore_cols, axis=1)
            if select_cols:
                select_cols = check_select_cols(select_cols)
                train_data_df = train_data_df[select_cols]
                test_data_df = test_data_df[select_cols]
                valid_data_df = valid_data_df[select_cols]
            if return_as_dataframe:
                return train_data_df, valid_data_df, test_data_df
            else:
                return convert_df_tolist(train_data_df, valid_data_df, test_data_df)
        elif output_label is not None:
            if header is None:
                if output_label is True:
                    column_drop = len(train_data_df.columns) - 1
                else:
                    column_drop = []
                    if len(output_label[0]) > 1:
                        for each_range in output_label:
                            column_drop = list(range(each_range[0] - 1, each_range[1]))
                    else:
                        column_drop = [output_label[0][0] - 1]
                if ignore_cols:
                    ignore_cols = check_ignore_cols(ignore_cols)
                    train_data_df = train_data_df.drop(ignore_cols, axis=1)
                    test_data_df = test_data_df.drop(ignore_cols, axis=1)
                    valid_data_df = valid_data_df.drop(ignore_cols, axis=1)
                train_label_df = train_data_df[column_drop].apply(literal_eval) if label_vector else train_data_df[
                    column_drop]
                train_data_df = train_data_df.drop(column_drop, axis=1)
                valid_label_df = valid_data_df[column_drop].apply(literal_eval) if label_vector else valid_data_df[
                    column_drop]
                valid_data_df = valid_data_df.drop(column_drop, axis=1)
                test_label_df = test_data_df[column_drop].apply(literal_eval) if label_vector else test_data_df[
                    column_drop]
                test_data_df = test_data_df.drop(column_drop, axis=1)
                if select_cols:
                    select_cols = check_select_cols(select_cols)
                    train_data_df = train_data_df[select_cols]
                    test_data_df = test_data_df[select_cols]
                    valid_data_df = valid_data_df[select_cols]
            else:
                column_drop = df.columns[-1] if output_label is True else output_label
                if ignore_cols:
                    ignore_cols = check_ignore_cols(ignore_cols)
                    train_data_df = train_data_df.drop(ignore_cols, axis=1)
                    test_data_df = test_data_df.drop(ignore_cols, axis=1)
                    valid_data_df = valid_data_df.drop(ignore_cols, axis=1)
                train_label_df = train_data_df[column_drop].apply(literal_eval) if label_vector else train_data_df[
                    column_drop]
                train_data_df = train_data_df.drop(column_drop, axis=1)
                valid_label_df = valid_data_df[column_drop].apply(literal_eval) if label_vector else valid_data_df[
                    column_drop]
                valid_data_df = valid_data_df.drop(column_drop, axis=1)
                test_label_df = test_data_df[column_drop].apply(literal_eval) if label_vector else test_data_df[
                    column_drop]
                test_data_df = test_data_df.drop(column_drop, axis=1)
                if select_cols:
                    select_cols = check_select_cols(select_cols)
                    train_data_df = train_data_df[select_cols]
                    test_data_df = test_data_df[select_cols]
                    valid_data_df = valid_data_df[select_cols]
            if return_as_dataframe:
                return train_data_df, train_label_df, valid_data_df, valid_label_df, test_data_df, test_label_df
            else:
                return convert_df_tolist(train_data_df, train_label_df, valid_data_df,
                                         valid_label_df, test_data_df, test_label_df)
    else:
        raise Exception("Length of split_ratio should be 3 with sum of elements equal to 100")


@typechecked()
def check_select_cols(select_cols: list = None):
    if isinstance(select_cols[0], str):
        return select_cols
    else:
        if isinstance(select_cols[0], list):
            col_range = []
            for each_range in select_cols:
                col_range += list(range(each_range[0] - 1, each_range[1]))
        return col_range


@typechecked()
def check_ignore_cols(ignore_cols: list = None):
    if isinstance(ignore_cols[0], str):
        return ignore_cols
    else:
        if isinstance(ignore_cols[0], list):
            col_range = []
            for each_range in ignore_cols:
                col_range += list(range(each_range[0] - 1, each_range[1]))
        return col_range


@typechecked()
def read_network(network_name: str, layer_data: dict, split_ratio: Union[list, None], normalize: bool = False,
                 randomize: bool = False,
                 output_label: bool = False):
    """
    | **@author:** Prathyush SP
    |
    | Function used to read data from a Tensorflow Saved session
    :param network_name: Network name with Layer Name [ex: network_name.layer_name]
    :param layer_data: Data to feed to saved network placeholders
    :param split_ratio: Split Ratio
    :param normalize: Normalize the data
    :param randomize: Randomize the data
    :param output_label: Output Label
    :return: Tapped Values spit based on split ratio
    """
    from rztdl.dl.network import Prediction
    if '.' in network_name:
        if layer_data:
            if len(network_name.split('.')) == 2:
                network_name, layer_name = network_name.split('.')
                prediction = Prediction(network_name=network_name).predict(layer_name=layer_name, data=layer_data)
                shape = list(prediction[0].shape)
                prediction, shape[0] = \
                    [np.reshape(pr, [shape[0], functools.reduce(operator.mul, shape[1:], 1)]) for pr in prediction][
                        0], 1
                return [[np.reshape(sample, newshape=shape[1:]) for sample in d] for d in
                        read_csv(data=prediction, split_ratio=split_ratio, normalize=normalize, randomize=randomize,
                                 output_label=output_label)]
            else:
                raise Exception('Layer Input supports: network.layer_name. Given: {}'.format(network_name))
        else:
            raise Exception('Layer Data is None')
    else:
        raise Exception('Layer Input supports: network.layer_name. Given: {}'.format(network_name))


def read_hdfs(filename, host, split_ratio, delimiter=',', normalize=False, dtype=None, header=None, skiprows=None,
              index_col=False, output_label=True, randomize=False, return_as_dataframe=False, describe=False,
              label_vector=False):
    client = HdfsClient(hosts=host)
    return read_csv(filename=client.open(filename), split_ratio=split_ratio, delimiter=delimiter, normalize=normalize,
                    dtype=dtype,
                    header=header, skiprows=skiprows, index_col=index_col, output_label=output_label,
                    randomize=randomize, return_as_dataframe=return_as_dataframe, describe=describe,
                    label_vector=label_vector)


@typechecked()
def to_csv(data: Union[dict, list, np.ndarray], save_path: str, header: bool = True, index: bool = False):
    """
    | **@author:** Prathyush SP
    |
    | Write to CSV File
    :param data: Data
    :param save_path: Save Path
    :param header: Header
    :param index: Index
    | ..todo ::
        Prathyush SP:
        1. Fix Logging Issue
    """
    if isinstance(data, dict):
        keys, vals = [], []
        for k, v in data.items():
            keys.append(k)
            vals.append(v)
        vals = list(zip(*vals))
        pd.DataFrame(data=vals, columns=keys).to_csv(path_or_buf=save_path, header=header, index=index)
    else:
        pd.DataFrame(data=data).to_csv(path_or_buf=save_path, header=header, index=index)
    logger.info('CSV Generated at ' + save_path)
    print('CSV Generated at', save_path)
