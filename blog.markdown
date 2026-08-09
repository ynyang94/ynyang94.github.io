---
layout: single
title: "Blog Posts"
permalink: /blog/
author_profile: true
---

Research notes on optimization and machine learning.

<div class="blog-index">
{% for post in site.posts %}
  {% if post.featured_blog %}
  <article class="blog-index__item">
    <p class="blog-index__date">{{ post.date | date: "%B %-d, %Y" }}</p>
    <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    {% if post.excerpt %}<p>{{ post.excerpt | strip_html | strip_newlines }}</p>{% endif %}
    <a class="blog-index__link" href="{{ post.url | relative_url }}">Read the article &rarr;</a>
  </article>
  {% endif %}
{% endfor %}
</div>
