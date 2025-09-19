---
layout: page
title: Publications
---

{% assign sorted_pubs = site.publications | sort: 'year' | reverse %}
{% for pub in sorted_pubs %}
### [{{ pub.title }}]({{ pub.url | relative_url }}) ({{ pub.year }})
*{{ pub.venue }}*  
{{ pub.content | strip_html | truncatewords: 30 }}
{% endfor %}
