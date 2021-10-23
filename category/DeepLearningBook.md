---
layout: page
title: Deep Learning
categories: [level_LecturesAndBooks]
---


<h2>Object Detection</h2>
<ul class="related-posts">
    {% for post in site.tags.DeepLearningBook %}
        <h3>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <small>{{ post.date | date_to_string }}</small>
        </h3>
    {% endfor %}
</ul>
