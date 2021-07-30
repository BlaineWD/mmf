#!/bin/bash

mmf_run config=projects/hateful_memes/configs/unimodal/image.yaml \
    model=unimodal_image \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/unimodal/with_features.yaml \
    model=unimodal_image \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/unimodal/bert.yaml \
    model=unimodal_text \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/late_fusion/defaults.yaml \
    model=late_fusion \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/late_fusion/defaults.yaml \
    model=late_fusion \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml \
    model=mmbt \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/mmbt/with_features.yaml \
    model=mmbt \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/vilbert/defaults.yaml \
    model=vilbert \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/visual_bert/direct.yaml \
    model=visual_bert \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/vilbert/from_cc.yaml \
    model=vilbert \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true

mmf_run config=projects/hateful_memes/configs/visual_bert/from_coco.yaml \
    model=visual_bert \
    dataset=hateful_memes \
    training.max_updates=5000 \
    training.log_interval=50 \
    training.checkpoint_interval=5000 \
    training.evaluation_interval=500 \
    training.batch_size=16 \
    training.log_format=json \
    training.evaluate_metrics=true
