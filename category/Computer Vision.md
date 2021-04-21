---
layout: page
title: Computer Vision
---

하나씩 정리. 분류는 생각나는대로

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


<h2>Unclassified</h2>
<ul class="related-posts">
    {% for post in site.tags.Unclassified %}
        <h3>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <small>{{ post.date | date_to_string }}</small>
        </h3>
    {% endfor %}
</ul>