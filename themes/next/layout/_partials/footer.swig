<div class="copyright" >
{% set current = date(Date.now(), "YYYY") %}
&copy; {% if theme.since and theme.since != current %} {{ theme.since }} &mdash; {% endif %}
<span itemprop="copyrightYear">{{ current }}</span>
<span class="with-love">
<i class="fa fa-{{ theme.authoricon }}"></i>
</span>
<span class="author" itemprop="copyrightHolder">{{ config.author }}</span>
{% if theme.post_wordcount.totalcount %}
<span class="post-meta-divider">|</span>
<span class="post-meta-item-icon">
<i class="fa fa-area-chart"></i>
</span>
{% if theme.post_wordcount.item_text %}
<span class="post-meta-item-text">{{ __('post.totalcount') }}&#58;</span>
{% endif %}
<span title="{{ __('post.totalcount') }}">
{{ totalcount(site, '0,0.0a') }}
</span>
{% endif %}
</div>
{% if theme.copyright %}
</div>
<span class="post-meta-divider">|</span>
</div>
{% endif %}
<script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js">
</script>
<span id="busuanzi_container_site_pv">
    本站总访问量<i class="fa fa-eye"></i><span id="busuanzi_value_site_pv"></span>次
</span>
  
<span id="busuanzi_container_site_uv">
  本站访客数<i class="fa fa-user"></i><span id="busuanzi_value_site_uv"></span>人次
</span>