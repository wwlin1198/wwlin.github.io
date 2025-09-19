---
layout: page
---

# Willy Lin's Blog

Writing down my research journey.

## Posts

{% for post in site.posts %}
### [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}

<!-- DEBUG: Posts section ends here -->

## Publications

{% assign sorted_pubs = site.publications | sort: 'year' | reverse %}
{% for pub in sorted_pubs %}
### [{{ pub.title }}]({{ pub.url | relative_url }}) ({{ pub.year }})
*{{ pub.venue }}*  
{{ pub.content | strip_html | truncatewords: 30 }}
{% endfor %}