---
layout: page
title: Publications
---

## Publications and Papers

{% assign sorted_pubs = site.publications | sort: 'year' | reverse %}
{% for pub in sorted_pubs %}
**{{ pub.title }}** ({{ pub.year }})  
*{{ pub.venue }}*  
[Link]({{ pub.url }})

{% endfor %}
