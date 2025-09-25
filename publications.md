---
layout: page
title: Publications
---

{% if site.publications and site.publications.size > 0 %}
  {% assign sorted_pubs = site.publications | sort: 'year' | reverse %}
  {% for pub in sorted_pubs %}
### [{{ pub.title }}]({{ pub.url | relative_url }}) ({{ pub.year }})
*{{ pub.venue }}*  
{{ pub.content | strip_html | truncatewords: 250 }}
  {% endfor %}
{% else %}
  <p>No publications found.</p>
{% endif %}
