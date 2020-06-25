---
layout: page
---

<ul class="related-posts">
    {% for post in site.tags.papers %}
        <h3>
            <a href="{{ post.url }}">{{ post.title }}</a>
            <small>{{ post.date | date_to_string }}</small>
        </h3>
    {% endfor %}
</ul>