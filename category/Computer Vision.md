---
layout: page
title: Papers
categories: [level_top]
---

연도 순서대로 정리되어있지 않음. 

<h2>Object Detection</h2>
<ul class="related-posts">
    {% for post in site.tags.ObjectDetection %}
        <h3>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <small>{{ post.date | date_to_string }}</small>
        </h3>
    {% endfor %}
</ul>


<h2>Segmentation</h2>
<ul class="related-posts">
    {% for post in site.tags.Segmentation %}
        <h3>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <small>{{ post.date | date_to_string }}</small>
        </h3>
    {% endfor %}
</ul>

<h2>Face Recognition</h2>
<ul class="related-posts">
    {% for post in site.tags.FaceRecognition %}
        <h3>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <small>{{ post.date | date_to_string }}</small>
        </h3>
    {% endfor %}
</ul>

<h2>Clustering</h2>
<ul class="related-posts">
    {% for post in site.tags.Clustering %}
        <h3>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <small>{{ post.date | date_to_string }}</small>
        </h3>
    {% endfor %}
</ul>


<h2>GAN</h2>
<ul class="related-posts">
    {% for post in site.tags.GAN %}
        <h3>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <small>{{ post.date | date_to_string }}</small>
        </h3>
    {% endfor %}
</ul>


<h2>Model Architecture</h2>
<ul class="related-posts">
    {% for post in site.tags.ModelArchitecture %}
        <h3>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <small>{{ post.date | date_to_string }}</small>
        </h3>
    {% endfor %}
</ul>